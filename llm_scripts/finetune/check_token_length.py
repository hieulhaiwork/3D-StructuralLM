import json
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm


FILE_PATH = "datasets/generated_dataset_1500_10500.jsonl"
MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

print(f"Loading Tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


print(f"Reading {FILE_PATH}...")
dataset = load_dataset("json", data_files=FILE_PATH, split="train")


print("measuring token lengths (including system prompts + json overhead)...")

lengths = []

for row in tqdm(dataset):
    messages = row["messages"]

    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    tokens = tokenizer.encode(formatted_text)
    lengths.append(len(tokens))

max_len = max(lengths)
avg_len = np.mean(lengths)
p95_len = np.percentile(lengths, 95)

print("\n" + "=" * 40)
print("TOKEN STATISTICS")
print("=" * 40)
print(f"Total Examples:      {len(lengths)}")
print(f"Average Length:      {int(avg_len)} tokens")
print(f"95th Percentile:     {int(p95_len)} tokens")
print(f"Maximum Length:      {max_len} tokens")
print("=" * 40)

possible_sizes = [2048, 4096, 8192, 16384, 32768]
recommended = 8192

for size in possible_sizes:
    if size >= max_len:
        recommended = size
        break

print(f"\nRECOMMENDED SETTING:")
print(f"MAX_SEQ_LENGTH = {recommended}")
print("=" * 40)

if max_len > 8192:
    print("Warning: Your data is very long. Ensure your GPU can handle it.")
