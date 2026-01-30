import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_TRAIN_FILES = [
    "train/sft_train_agents_eng.jsonl",
    "train/sft_train_agents_rus.jsonl",
    "train/sft_train_agents_temp_03_big.jsonl",
]


def format_example(example: dict[str, Any]) -> dict[str, str]:
    input_text = example["input"]
    output_value = example["output"]
    if not isinstance(output_value, str):
        output_value = json.dumps(output_value, ensure_ascii=False)
    text = f"### Input\n{input_text}\n\n### Output\n{output_value}"
    return {"text": text}


def pick_precision() -> tuple[torch.dtype, bool, bool]:
    bf16_supported = False
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = False

    if bf16_supported:
        return torch.bfloat16, False, True
    return torch.float16, True, False


def build_lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(model_name: str, torch_dtype: torch.dtype) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str, default=DEFAULT_MODEL)

    p.add_argument(
        "--train_files",
        nargs="+",
        default=DEFAULT_TRAIN_FILES,
        help="Список jsonl файлов (в каждом ожидаются поля input/output).",
    )

    p.add_argument("--output_dir", type=str, default="output/qwen3-4b-lora-sft")

    p.add_argument("--max_seq_length", type=int, default=2048)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=3.0)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")

    p.add_argument("--seed", type=int, default=42)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--report_to", type=str, default="none")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch_dtype, use_fp16, use_bf16 = pick_precision()

    tokenizer = load_tokenizer(args.model_name_or_path)

    dataset = load_dataset("json", data_files=args.train_files, split="train")
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=args.report_to,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    lora_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    model = load_model(args.model_name_or_path, torch_dtype=torch_dtype)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    del trainer, model, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
