"""QLoRA SFT v2 of Qwen3-4B-Instruct-2507 on whole tool trajectories (Unsloth, 4-bit base, LoRA r16 on attention and MLP).

    <venv with unsloth>/bin/python scripts/train_agent_sft_v2.py --data-dir outputs/sft/data_v2 --out outputs/sft/s2_01 \
        --model models/qwen3-4b-instruct-2507 --max-len 16384 --epochs 2 --lr 1e-4

v1 expanded every assistant decision into its own example, so the ~5-10k-token prompt (tool schemas, history) was trained once
per decision; here one trajectory is one example and the loss covers every assistant turn (tool calls and the final stop turn),
with the LM head computed only from the first assistant token on (CompletionLogitsTrainer). The text is the checkpoint's own chat
template; before training, every assistant turn's prefix is checked to equal what inference sends (add_generation_prompt=True)."""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

from unsloth import FastLanguageModel  # noqa: I001  (unsloth patches transformers on import: keep it first)
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

SPAN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def encode(row: dict, tok, check: bool) -> dict:
    text = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
    spans = [(m.start(1), m.end(1)) for m in SPAN.finditer(text)]
    if len(spans) != sum(m["role"] == "assistant" for m in row["messages"]):
        raise ValueError("assistant spans do not match the assistant messages")
    if check:
        k = 0
        for i, m in enumerate(row["messages"]):
            if m["role"] != "assistant":
                continue
            prefix = tok.apply_chat_template(row["messages"][:i], tools=row["tools"], tokenize=False, add_generation_prompt=True)
            if text[:spans[k][0]] != prefix:
                raise ValueError(f"training text differs from the inference prefix before assistant turn {k}")
            k += 1
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels, j = [-100] * len(ids), 0
    for t, (s, e) in enumerate(offs):
        while j < len(spans) and s >= spans[j][1]:
            j += 1
        if j < len(spans) and s >= spans[j][0] and e <= spans[j][1] and e > s:
            labels[t] = ids[t]
    return {"input_ids": ids, "labels": labels}


class CompletionLogitsTrainer(Trainer):
    """Same loss; the LM head runs only from the position before the first trained token (most tokens are the prompt)."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        first = (labels != -100).any(0).nonzero()
        if len(first):
            m = labels.shape[1] - int(first[0]) + 1
            inputs = {**inputs, "labels": labels[:, -m:].contiguous(), "logits_to_keep": m}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="models/qwen3-4b-instruct-2507")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1, help="per-device batch; rows are grouped by length to limit padding")
    ap.add_argument("--save-steps", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--init-adapter", type=Path)
    ap.add_argument("--limit-rows", type=int)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(model_name=a.model, max_seq_length=a.max_len, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(model, r=a.r, lora_alpha=a.alpha, lora_dropout=0, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                             use_gradient_checkpointing="unsloth", random_state=a.seed)
    if a.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(model, load_file(a.init_adapter / "adapter_model.safetensors"))
        print(f"continuing from {a.init_adapter}: {len(res.unexpected_keys)} unexpected keys")
    raw = [json.loads(l) for l in (a.data_dir / "train.jsonl").open(encoding="utf-8")]
    rows, dropped = [], 0
    for i, r in enumerate(raw):
        e = encode(r, tok, check=i < 60)
        if len(e["input_ids"]) > a.max_len:
            dropped += 1
            continue
        rows.append(e)
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.limit_rows] if a.limit_rows else rows
    n_tok = sum(len(r["input_ids"]) for r in rows)
    n_lab = sum(sum(x != -100 for x in r["labels"]) for r in rows)
    print(f"{len(rows)} trajectories ({dropped} over {a.max_len} tokens dropped); tokens {n_tok:,}, trained {n_lab:,} ({n_lab / n_tok:.1%})", flush=True)
    bf16 = torch.cuda.is_bf16_supported()
    trainer = CompletionLogitsTrainer(
        model=model, train_dataset=Dataset.from_list(rows),
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        args=TrainingArguments(output_dir=str(a.out), per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum, train_sampling_strategy="group_by_length" if a.batch > 1 else "random",
                               num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, warmup_steps=10,
                               lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.0, bf16=bf16, fp16=not bf16,
                               logging_steps=5, save_strategy="steps", save_steps=a.save_steps, save_total_limit=4,
                               report_to="none", seed=a.seed, remove_unused_columns=False))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "lifemate_training_config.json").write_text(json.dumps({**vars(a), "data_dir": str(a.data_dir), "out": str(a.out),
        "init_adapter": str(a.init_adapter) if a.init_adapter else None, "trajectories": len(rows), "dropped_over_max_len": dropped,
        "tokens": n_tok, "trained_tokens": n_lab}, indent=2) + "\n")
    t = time.time()
    last = max(a.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]), default=None) if a.resume else None
    trainer.train(resume_from_checkpoint=str(last) if last else None)
    dt = time.time() - t
    print(f"{trainer.state.global_step} steps in {dt:.0f}s; peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    model.save_pretrained(str(a.out / "adapter"))
    tok.save_pretrained(str(a.out / "adapter"))


if __name__ == "__main__":
    main()
