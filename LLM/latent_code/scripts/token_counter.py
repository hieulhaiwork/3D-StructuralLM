from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

dataset_path = "latent_dataset_for_finetuning_v1_tiny_twoshot.json"   # or "alpaca_data.jsonl"
model_name = "Qwen/Qwen3-8B"

# Load dataset
dataset = load_dataset("json", data_files=dataset_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to tokenize and get length
def get_length(example):
    text_parts = []
    if "instruction" in example:
        text_parts.append(example["instruction"])
    if "input" in example and example["input"]:
        text_parts.append(example["input"])
    if "output" in example:
        text_parts.append(example["output"])
    text = "\n".join(text_parts).strip()
    return {"length": len(tokenizer.encode(text, add_special_tokens=True))}

# Apply to dataset
dataset = dataset.map(get_length)

# Collect lengths
lengths = dataset["train"]["length"]

print("Number of samples:", len(lengths))
print("Max token length:", max(lengths))
print("95th percentile:", int(np.percentile(lengths, 95)))
print("99th percentile:", int(np.percentile(lengths, 99)))
