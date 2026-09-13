from unsloth import FastLanguageModel

ADAPTER_DIR = "models/adapters/llama3.1_8b_structuralm_128lora_v2"
OUTPUT_DIR = "structuralm_merged_16bit"
MAX_SEQ_LENGTH = 8192

print("Loading model for merging...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    dtype=None,
    device_map="cpu",
)

print("Merging LoRA into Base Model (16-bit)...")
model.save_pretrained_merged(
    OUTPUT_DIR,
    tokenizer,
    save_method="merged_16bit",
)

print(f"Merged model saved to: {OUTPUT_DIR}")
