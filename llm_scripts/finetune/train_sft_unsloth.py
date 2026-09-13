import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template


MODEL_SLUG = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

OUTPUT_DIR = "models/llama3.1_8b_structuralm_128lora_v2"
DATA_FILE = "datasets/v1/generated_dataset_1500_10500.jsonl"
MAX_SEQ_LENGTH = 8192

RANK = 128

print(f"Loading {MODEL_SLUG}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_SLUG,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

dataset = load_dataset("json", data_files=DATA_FILE, split="train")

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)


def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)

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

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        learning_rate=5.0e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
    ),
)

print(f"Starting BIG LORA (r={RANK}) Training on H100...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved LoRA adapters to {OUTPUT_DIR}")
