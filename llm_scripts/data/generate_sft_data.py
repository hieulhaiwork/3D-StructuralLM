import os
import json
import time
import random
import gc
from typing import Dict, Any
from pathlib import Path
from tqdm import tqdm

from Multi_agents.multi_agents import get_llm, run_multi_agents_scene, SceneState


NUM_SAMPLES = 20000
OUTPUT_FILE = "/home/quanghm5/structuralm/temp/generated_data_v2.jsonl"
DELAY_BETWEEN_CALLS = 0.5


def generate_synthetic_prompt() -> str:
    room_types = [
        "bedroom",
        "living room",
        "kitchen",
        "study",
        "minimalist office",
        "dining room",
        "room",
    ]

    selected_room = random.choice(room_types)

    system_prompt = (
        "You are a creative writer generating prompts for 3D scene synthesis."
    )

    user_prompt = f"""
    Generate a SINGLE furniture description of a {selected_room}. Style: simple and direct.
    Constraints: 
    - Mention 3 to 4 distinct objects.
    - Keep it under 40 words.
    - Output ONLY the raw description text. No numbering, no quotes"""
    try:
        llm = get_llm()
        resp = llm.invoke([("system", system_prompt), ("user", user_prompt)])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return content.strip().replace('"', "").replace("\n", " ")
    except Exception as e:
        print(f"Error generating prompt: {e}")
        return None


def format_trace_line(agent_name: str, trace: Dict[str, Any]) -> str:
    """
    Formats a single trace into a JSONL line string.
    Reconstructs exact inputs/outputs used during inference.
    """
    input_data = trace["input"]
    output_data = trace["output"]
    system_content = trace["system_prompt"]
    user_content = ""
    assistant_content = ""

    if agent_name == "planner":
        user_content = input_data["effective_prompt"]
    elif agent_name == "entity_extraction":
        user_content = f"User prompt:\n{input_data['prompt']}"
    elif agent_name == "attribute_enrichment":
        user_content = input_data["user_prompt"]
    elif agent_name == "relational_inference":
        user_content = json.dumps(input_data, indent=2)
    elif agent_name == "attribute_extraction":
        rec_input = {
            "prompt": input_data["prompt"],
            "nodes": input_data["nodes_before"],
        }
        user_content = json.dumps(rec_input, indent=2, ensure_ascii=False)
    elif agent_name == "size_scale_estimation":
        rec_input = {
            "prompt": input_data["prompt"],
            "blueprint": input_data["blueprint_before"],
            "anchor": input_data["anchor"],
        }
        user_content = json.dumps(rec_input, indent=2, ensure_ascii=False)
    elif agent_name == "graph_formalization":
        rec_input = {"blueprint": input_data["blueprint_before"]}
        user_content = json.dumps(rec_input, indent=2, ensure_ascii=False)

    if agent_name == "planner":
        assistant_content = output_data["plan_notes"]
    elif agent_name == "attribute_enrichment":
        assistant_content = output_data["enriched_prompt"]
    elif agent_name == "entity_extraction":
        assistant_content = json.dumps({"nodes": output_data["nodes"]}, indent=2)
    elif agent_name == "relational_inference":
        assistant_content = json.dumps({"edges": output_data["edges"]}, indent=2)
    elif agent_name == "attribute_extraction":
        assistant_content = json.dumps({"nodes": output_data["nodes_after"]}, indent=2)
    elif agent_name == "size_scale_estimation":
        assistant_content = json.dumps(output_data["blueprint_after"], indent=2)
    elif agent_name == "graph_formalization":
        assistant_content = json.dumps(output_data["scene_graph"], indent=2)

    record = {
        "agent": agent_name,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }

    return json.dumps(record, ensure_ascii=False)


if __name__ == "__main__":
    print(f"Starting Lazy Generation: {NUM_SAMPLES} samples")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "a", encoding="utf8") as f_out:

        pbar = tqdm(total=NUM_SAMPLES, desc="Generating Scenes")

        for i in range(NUM_SAMPLES):
            try:
                prompt_text = generate_synthetic_prompt()

                result = run_multi_agents_scene(prompt_text, enrich_scene=True)
                traces = result.get("traces", {})

                if traces:
                    for agent_name, trace_data in traces.items():
                        try:
                            jsonl_line = format_trace_line(agent_name, trace_data)
                            f_out.write(jsonl_line + "\n")
                        except Exception as fmt_err:
                            print(f"[!] Format error {agent_name}: {fmt_err}")

                    f_out.flush()

                pbar.update(1)

                del result
                del traces
                gc.collect()

                time.sleep(DELAY_BETWEEN_CALLS)

            except Exception as e:
                print(f"\n[!] Pipeline error on sample {i}: {e}")
                continue

        pbar.close()

    print(f"\nFinished. Data saved incrementally to: {os.path.abspath(OUTPUT_FILE)}")
