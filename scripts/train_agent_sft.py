"""Offline QLoRA SFT of Qwen3-4B-Instruct-2507 for E1 tool decisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def decisions(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        prefix = []
        for message in row["messages"]:
            if message["role"] == "assistant":
                result.append({"prompt": prefix.copy(), "completion": [message], "tools": row["tools"]})
            prefix.append(message)
    return result


def patched_tokenizer(model_dir: Path, sample: dict):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    original = tokenizer.apply_chat_template(sample["prompt"] + sample["completion"], tools=sample["tools"], tokenize=False)
    # Mark the assistant branch in the checkpoint's own template. This preserves
    # its exact Qwen3-2507 tool-call serialization and only adds loss boundaries.
    template = tokenizer.chat_template
    opening = """{%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\\n' + content }}"""
    closing = """        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}"""
    if template.count(opening) != 1 or template.count(closing) != 1:
        raise RuntimeError("Unexpected Qwen3-2507 chat template; cannot add assistant loss mask safely")
    tokenizer.chat_template = template.replace(opening, opening.replace("{{-", "{% generation %}{{-", 1), 1).replace(
        closing, closing.replace("\n    {%- elif", "{% endgeneration %}\n    {%- elif"), 1)
    patched = tokenizer.apply_chat_template(sample["prompt"] + sample["completion"], tools=sample["tools"], tokenize=False)
    if patched != original:
        raise RuntimeError("TRL assistant-mask template changes Qwen inference rendering")
    tokenized = tokenizer.apply_chat_template(sample["prompt"] + sample["completion"], tools=sample["tools"],
                                              tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    if not any(tokenized["assistant_masks"]):
        raise RuntimeError("No assistant tokens were marked for supervised loss")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/agent_sft_qlora.json")
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "outputs/sft/data_v1")
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models/qwen3-4b-instruct-2507")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs/sft/qwen3_4b_e1_qlora_01")
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-decisions", type=int, help="limit CPU preflight in --dry-run only")
    preliminary, _ = parser.parse_known_args()
    parser.set_defaults(**json.loads(preliminary.config.read_text(encoding="utf-8")))
    args = parser.parse_args()
    if args.max_decisions and not args.dry_run:
        parser.error("--max-decisions is only for --dry-run")
    if args.max_length < 1024 or args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("invalid batch size, accumulation or sequence length")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not args.dry_run:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible and visible != str(args.physical_gpu):
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible}; expected physical GPU {args.physical_gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    model_dir = args.model_dir.resolve()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    validation_bytes = (PROJECT / "outputs/splits/validation_ids.json").read_bytes()
    forbidden = set(json.loads(validation_bytes))
    if (manifest["validation_excluded"] != 200 or len(forbidden) != 200 or
            manifest["validation_ids_sha256"] != hashlib.sha256(validation_bytes).hexdigest()):
        raise ValueError("SFT data manifest does not exclude the 200 validation IDs")
    train_rows, dev_rows = read_rows(args.data_dir / "train.jsonl"), read_rows(args.data_dir / "dev.jsonl")
    if not train_rows or not dev_rows:
        raise ValueError("SFT train and dev datasets must both be nonempty")
    train_ids = {row["sample_id"] for row in train_rows}
    dev_ids = {row["sample_id"] for row in dev_rows}
    if (len(train_ids) != len(train_rows) or len(dev_ids) != len(dev_rows) or
            train_ids & dev_ids or (train_ids | dev_ids) & forbidden):
        raise ValueError("SFT data duplicates IDs or overlaps the frozen validation split")
    train, dev = decisions(train_rows), decisions(dev_rows)
    if args.max_decisions:
        train, dev = train[:args.max_decisions], dev[:max(1, min(args.max_decisions, len(dev)))]
    tokenizer = patched_tokenizer(model_dir, train[0])
    def within_limit(rows: list[dict]) -> tuple[list[dict], int]:
        kept = []
        for row in rows:
            length = len(tokenizer.apply_chat_template(row["prompt"] + row["completion"], tools=row["tools"], tokenize=True))
            if length <= args.max_length:
                kept.append(row)
        return kept, len(rows) - len(kept)
    train, train_dropped = within_limit(train)
    dev, dev_dropped = within_limit(dev)
    if not train or not dev:
        raise ValueError("no training/evaluation decisions fit --max-length")
    summary = {"train_decisions": len(train), "dev_decisions": len(dev), "train_too_long": train_dropped,
               "dev_too_long": dev_dropped, "max_length": args.max_length,
               "assistant_only_loss": True, "completion_only_loss": True}
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device for QLoRA")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 18 * 1024**3:
        raise RuntimeError(f"Only {free_bytes / 1024**3:.1f} GiB free on GPU {args.physical_gpu}; stop the vLLM server before training")
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True,
        quantization_config=quantization, device_map={"": 0}, dtype=dtype)
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    config = SFTConfig(output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation,
        max_length=args.max_length, assistant_only_loss=True, completion_only_loss=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        optim="paged_adamw_8bit", eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=3, logging_steps=10, report_to="none", seed=42,
        eos_token="<|im_end|>", packing=False)
    train_dataset = Dataset.from_list(train, on_mixed_types="use_json")
    dev_dataset = Dataset.from_list(dev, on_mixed_types="use_json")
    trainer = SFTTrainer(model=model, args=config, train_dataset=train_dataset, eval_dataset=dev_dataset,
                         processing_class=tokenizer, peft_config=lora,
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lifemate_training_config.json").write_text(json.dumps({**summary,
        "base_model": str(model_dir), "data_dir": str(args.data_dir.resolve()),
        "qlora": {"nf4": True, "rank": 16, "alpha": 32, "compute_dtype": str(dtype)},
        "learning_rate": args.learning_rate, "epochs": args.epochs}, indent=2) + "\n")
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    best_dir = args.output_dir / "best_adapter"
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    trainer.save_state()
    print(json.dumps({"best_adapter": str(best_dir), "best_checkpoint": trainer.state.best_model_checkpoint,
                      "best_eval_loss": trainer.state.best_metric}, indent=2))


if __name__ == "__main__":
    main()
