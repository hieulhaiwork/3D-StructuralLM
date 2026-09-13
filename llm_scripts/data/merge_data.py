import os

input_files = [
    "/home/quanghm5/training_llm/datasets/dpo_v1/dpo_data.jsonl",
    "/home/quanghm5/training_llm/datasets/dpo_v1/dpo_data2.jsonl",
    "/home/quanghm5/training_llm/datasets/dpo_v1/dpo_data3.jsonl",
]
output_file = (
    "/home/quanghm5/training_llm/datasets/dpo_v1/dpo_data_teacher_only_all.jsonl"
)

print(f"Merging {len(input_files)} files...")

line_count = 0

with open(output_file, "w", encoding="utf-8") as outfile:
    for fname in input_files:
        if os.path.exists(fname):
            print(f"  Reading {fname}...")
            with open(fname, "r", encoding="utf-8") as infile:
                for line in infile:
                    clean_line = line.strip()
                    if clean_line:
                        outfile.write(clean_line + "\n")
                        line_count += 1
        else:
            print(f"Warning: {fname} not found.")

print(f"Done! Created '{output_file}' with {line_count} total examples.")
