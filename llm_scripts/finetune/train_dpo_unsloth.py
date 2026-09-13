import os
import torch
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from transformers import TrainingArguments

PatchDPOTrainer()

SFT_ADAPTER_PATH = "models/adapters/llama3.1_8b_structuralm_128lora_v2"
OUTPUT_DIR = "models/adapters/llama3.1_8b_structuralm_128lora_sft_dpo_v2"
DATA_FILE = "datasets/dpo_v1/dpo_data_all.jsonl"

LEARNING_RATE = 5e-6
RANK = 128

print(f"Loading SFT Checkpoint from: {SFT_ADAPTER_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_ADAPTER_PATH,
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=256,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)


def format_dpo_func(example):
    prompt_messages = [
        {"from": "system", "value": example["system"]},
        {"from": "human", "value": example["prompt"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    chosen_text = example["chosen"]
    rejected_text = example["rejected"]

    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


print("Loading and Formatting Dataset...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")
dataset = dataset.map(format_dpo_func)


training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    beta=0.1,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    seed=3407,
    save_strategy="steps",
    save_steps=50,
    max_prompt_length=2048,
    max_length=4096,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

print("Starting Unsloth DPO Training...")
trainer.train()

print(f"Saving DPO-Tuned Adapter to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


import json

log_history = trainer.state.log_history

log_file_path = os.path.join(OUTPUT_DIR, "training_dpo_logs.json")

with open(log_file_path, "w") as f:
    json.dump(log_history, f, indent=2)

print(f"Training logs exported to: {log_file_path}")
