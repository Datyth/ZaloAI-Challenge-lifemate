"""QLoRA SFT of Qwen3.5-4B (text layers only, non-thinking replies) on whole tool trajectories; same data and loss as
train_agent_sft_v2.py. Rows go through lifemate.llm.merge_system, as at inference with LIFEMATE_MERGE_SYSTEM=1.
    /home/nhantd/projects/zalo-ai-challenge-datamate-26/.venv-train/bin/python scripts/train_agent_sft_q35.py \
        --data-dir outputs/sft/data_lao_base --out outputs/sft/q35_lao_base --model models/qwen3.5-4b --epochs 2 --lr 1e-4

Original notes (v2):

    <venv with unsloth>/bin/python scripts/train_agent_sft_v2.py --data-dir outputs/sft/data_v2 --out outputs/sft/s2_01 \
        --model models/qwen3-4b-instruct-2507 --max-len 16384 --epochs 2 --lr 1e-4

v1 expanded every assistant decision into its own example, so the ~5-10k-token prompt (tool schemas, history) was trained once
per decision; here one trajectory is one example and the loss covers every assistant turn (tool calls and the final stop turn),
with the LM head computed only from the first assistant token on (CompletionLogitsTrainer). The text is the checkpoint's own chat
template; before training, every assistant turn's prefix is checked to equal what inference sends (add_generation_prompt=True)."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unsloth import FastModel  # noqa: I001  (unsloth patches transformers on import: keep it first)
import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

FUNC = re.compile(r"<function=([^>\n]+)>")
PARAM = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)
SPAN = re.compile(r"<\|im_start\|>assistant\n(?:<think>\n\n</think>\n\n)?(.*?<\|im_end\|>)", re.S)  # empty think block is prompt


def encode(row: dict, tok, check: bool, fuzzy_weight: float | None = None) -> dict:
    from lifemate.llm import merge_system
    row = {**row, "messages": merge_system(row["messages"])}
    text = tok.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False, enable_thinking=False)
    spans = [(m.start(1), m.end(1)) for m in SPAN.finditer(text)]
    assistants = [m for m in row["messages"] if m["role"] == "assistant"]
    if len(spans) != len(assistants):
        raise ValueError("assistant spans do not match the assistant messages")
    # an assistant message with "train": false (an injected failed attempt, scripts/inject_recovery.py) stays in the context
    # but gets no loss: the model learns what to do after such a call, never to make it
    trained = [m.get("train", True) is not False for m in assistants]
    if check:
        k = 0
        for i, m in enumerate(row["messages"]):
            if m["role"] != "assistant":
                continue
            prefix = tok.apply_chat_template(row["messages"][:i], tools=row["tools"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if text[:spans[k][0]] != prefix:
                raise ValueError(f"training text differs from the inference prefix before assistant turn {k}")
            k += 1
    fuzzy = []   # character ranges of free-text argument values (x-match fuzzy: not hashed, scored by token F1 >= 0.5)
    if fuzzy_weight is not None:
        fz = {t["function"]["name"]: {k for k, p in t["function"]["parameters"].get("properties", {}).items()
                                      if isinstance(p, dict) and p.get("x-match") == "fuzzy"} for t in row["tools"]}
        for s0, e0 in spans:
            body = text[s0:e0]
            f = FUNC.search(body)
            keys = fz.get(f.group(1), set()) if f else set()
            fuzzy += [(s0 + p.start(2), s0 + p.end(2)) for p in PARAM.finditer(body) if p.group(1) in keys]
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    labels, weights, j = [-100] * len(ids), [0.0] * len(ids), 0
    for t, (s, e) in enumerate(offs):
        while j < len(spans) and s >= spans[j][1]:
            j += 1
        if j < len(spans) and trained[j] and s >= spans[j][0] and e <= spans[j][1] and e > s:
            labels[t] = ids[t]
            weights[t] = fuzzy_weight if fuzzy_weight is not None and any(a <= s < b for a, b in fuzzy) else 1.0
    out = {"input_ids": ids, "labels": labels}
    if fuzzy_weight is not None:
        out["loss_weights"] = weights
    return out


class WeightedCollator:
    """Batch-1 friendly padding for input_ids / labels / loss_weights."""

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, feats: list[dict]) -> dict:
        n = max(len(f["input_ids"]) for f in feats)
        pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]
        return {"input_ids": torch.tensor(pad([f["input_ids"] for f in feats], self.pad_id)),
                "attention_mask": torch.tensor(pad([[1] * len(f["input_ids"]) for f in feats], 0)),
                "labels": torch.tensor(pad([f["labels"] for f in feats], -100)),
                "loss_weights": torch.tensor(pad([f["loss_weights"] for f in feats], 0.0))}


class WeightedLossTrainer(Trainer):
    """Token-weighted cross entropy: the model returns hidden states only at the positions that predict a trained token
    (UNSLOTH_RETURN_HIDDEN_STATES=1, logits_to_keep = those indices), the LM head runs on them, and each token's loss is scaled
    by loss_weights. Normalised by the number of trained tokens (num_items_in_batch), like the unweighted loss, so weight-1
    tokens keep the same gradient scale as before and only the free-text tokens are turned down."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels, w = inputs["labels"], inputs["loss_weights"]
        assert labels.shape[0] == 1, "WeightedLossTrainer expects --batch 1"
        tgt_pos = (labels[0] != -100).nonzero()[:, 0]
        pos = tgt_pos - 1
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], logits_to_keep=pos)
        h = out.logits[0]                                  # hidden states at pos
        if h.shape[0] != pos.shape[0]:
            raise RuntimeError(f"expected {pos.shape[0]} hidden states, got {tuple(h.shape)}")
        head = model.get_output_embeddings()
        logits = head(h.to(head.weight.dtype)).float()
        ce = torch.nn.functional.cross_entropy(logits, labels[0, tgt_pos], reduction="none")
        denom = num_items_in_batch if num_items_in_batch is not None else torch.tensor(float(len(tgt_pos)))
        loss = (ce * w[0, tgt_pos].float()).sum() / denom.to(ce.device)
        return (loss, out) if return_outputs else loss


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
    ap.add_argument("--model", default="models/qwen3.5-4b")
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
    ap.add_argument("--bf16-base", action="store_true", help="LoRA on the bf16 base instead of 4-bit QLoRA")
    ap.add_argument("--grad-ckpt", choices=["unsloth", "on", "off"], default="unsloth", help="unsloth = checkpointing with CPU offload")
    ap.add_argument("--fuzzy-weight", type=float, help="loss weight of free-text argument values (x-match fuzzy); default: plain loss")
    a = ap.parse_args()

    # --bf16-base: LoRA on the bf16 base (what vLLM serves) instead of QLoRA on a 4-bit copy; faster on this card (no dequant)
    model, processor = FastModel.from_pretrained(model_name=a.model, max_seq_length=a.max_len, load_in_4bit=not a.bf16_base, full_finetuning=False)
    model = FastModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True,
                                     finetune_mlp_modules=True, r=a.r, lora_alpha=a.alpha, lora_dropout=0, bias="none",
                                     random_state=a.seed, use_gradient_checkpointing={"unsloth": "unsloth", "on": True, "off": False}[a.grad_ckpt])
    tok = getattr(processor, "tokenizer", processor)   # the processor wraps the text tokenizer
    if a.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(model, load_file(a.init_adapter / "adapter_model.safetensors"))
        print(f"continuing from {a.init_adapter}: {len(res.unexpected_keys)} unexpected keys")
    raw = [json.loads(l) for l in (a.data_dir / "train.jsonl").open(encoding="utf-8")]
    rows, dropped = [], 0
    for i, r in enumerate(raw):
        e = encode(r, tok, check=i < 60, fuzzy_weight=a.fuzzy_weight)
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
    if a.fuzzy_weight is not None:
        n_fz = sum(sum(x != 1.0 for x, l in zip(r["loss_weights"], r["labels"]) if l != -100) for r in rows)
        print(f"free-text tokens weighted {a.fuzzy_weight}: {n_fz:,} of {n_lab:,} trained tokens", flush=True)
    trainer_cls = WeightedLossTrainer if a.fuzzy_weight is not None else CompletionLogitsTrainer
    trainer = trainer_cls(
        model=model, train_dataset=Dataset.from_list(rows),
        data_collator=WeightedCollator(tok.pad_token_id) if a.fuzzy_weight is not None else DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
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
