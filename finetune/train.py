"""
Fine-tune SmolLM2-135M for B2B autocomplete using LoRA.

Trains on prefix->completion pairs from Alvoff search queries.
Exports merged model and converts to GGUF for llama.cpp deployment.

Usage:
  cd /Users/ron/Desktop/tipstat/sourcer/autocomplete
  uv run python finetune/train.py
"""

import logging
import os
import sys
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_LENGTH = 64


class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        log.info(f"Training started — {state.max_steps} total steps, {args.num_train_epochs} epochs")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        elapsed = time.time() - self.start_time
        pct = 100 * state.global_step / state.max_steps if state.max_steps > 0 else 0
        eta_s = (elapsed / max(state.global_step, 1)) * (state.max_steps - state.global_step)
        eta_m = eta_s / 60

        parts = [f"step {state.global_step}/{state.max_steps} ({pct:.0f}%)"]
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        parts.append(f"elapsed={elapsed/60:.1f}m")
        parts.append(f"eta={eta_m:.1f}m")

        log.info(" | ".join(parts))

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            log.info(f"  eval — loss={metrics.get('eval_loss', 0):.4f}, "
                     f"perplexity={2**metrics.get('eval_loss', 0):.2f}")

    def on_save(self, args, state, control, **kwargs):
        log.info(f"  checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        log.info(f"Training complete — {state.global_step} steps in {elapsed/60:.1f} minutes")


def main():
    log.info("=" * 60)
    log.info("SmolLM2-135M Fine-tuning for B2B Autocomplete")
    log.info("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info(f"Loading tokenizer and model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
    )
    log.info(f"Base model params: {sum(p.numel() for p in model.parameters()):,}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"LoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    log.info("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(DATA_DIR, "train.jsonl"),
            "validation": os.path.join(DATA_DIR, "val.jsonl"),
        },
    )
    log.info(f"Train: {len(dataset['train']):,} examples, Val: {len(dataset['validation']):,} examples")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    log.info("Tokenizing...")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        learning_rate=3e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[ProgressCallback()],
    )

    # Auto-resume from latest checkpoint if one exists
    checkpoints = sorted(
        [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
    ) if os.path.exists(OUTPUT_DIR) else []
    if checkpoints:
        log.info(f"Resuming from {checkpoints[-1]}")
    else:
        log.info("Starting training from scratch...")
    trainer.train(resume_from_checkpoint=bool(checkpoints))

    log.info("Saving LoRA adapter...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "lora-adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "lora-adapter"))

    log.info("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()
    merged_path = os.path.join(OUTPUT_DIR, "merged")
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    log.info(f"Merged model saved to {merged_path}")
    log.info("")
    log.info("Next steps:")
    log.info("  1. uv pip install gguf sentencepiece")
    log.info("  2. git clone --depth=1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp")
    log.info("  3. uv run python finetune/convert_to_gguf.py")


if __name__ == "__main__":
    main()
