import json
import os

INPUT_FILE = "datasets/generated_dataset_1500_10500.jsonl"

OUT_RELATIONAL = "datasets/dataset_relational_1500_15000.jsonl"
OUT_SCALE = "datasets/dataset_scale_1500_10500.jsonl"
OUT_GENERAL = "datasets/dataset_general_1500_10500.jsonl"

RELATIONAL_AGENTS = {"relational_inference"}

SCALE_AGENTS = {"size_scale_estimation"}


def split_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")

    counts = {"relational": 0, "scale": 0, "general": 0}

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, open(
        OUT_RELATIONAL, "w", encoding="utf-8"
    ) as f_rel, open(OUT_SCALE, "w", encoding="utf-8") as f_scale, open(
        OUT_GENERAL, "w", encoding="utf-8"
    ) as f_gen:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                agent_name = data.get("agent", "unknown")

                if agent_name in RELATIONAL_AGENTS:
                    f_rel.write(line + "\n")
                    counts["relational"] += 1

                elif agent_name in SCALE_AGENTS:
                    f_scale.write(line + "\n")
                    counts["scale"] += 1

                else:
                    f_gen.write(line + "\n")
                    counts["general"] += 1

            except json.JSONDecodeError:
                print("Skipping invalid JSON line")

    print("\nSplitting Complete!")
    print("=" * 40)
    print(
        f"1. Relational Model Data: {counts['relational']:>5} lines -> {OUT_RELATIONAL}"
    )
    print(f"2. Scale Model Data:      {counts['scale']:>5} lines -> {OUT_SCALE}")
    print(f"3. General Model Data:    {counts['general']:>5} lines -> {OUT_GENERAL}")
    print("=" * 40)
    print("Total lines processed:", sum(counts.values()))


if __name__ == "__main__":
    split_data()
