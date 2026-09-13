import os
import json
import time
import random
import copy
import gc
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm
from langchain_openai import ChatOpenAI

from Multi_agents.multi_agents import run_multi_agents_scene, _extract_json_str, PROMPTS

NUM_SAMPLES = 600
OUTPUT_FILE = "dpo_dataset/dpo_data_v1.jsonl"

STUDENT_API_KEY = "EMPTY"
STUDENT_BASE_URL = "http://localhost:8003/v1"
STUDENT_MODEL_NAME = "structuralm_v1"

JUDGE_API_KEY = os.getenv("MY_LLM_API_KEY")
JUDGE_BASE_URL = os.getenv("MY_LLM_BASE_URL")
JUDGE_MODEL_NAME = os.getenv("SCENE_MODEL_NAME")


def get_student_llm(temperature=0.7) -> ChatOpenAI:
    return ChatOpenAI(
        model=STUDENT_MODEL_NAME,
        api_key=STUDENT_API_KEY,
        base_url=STUDENT_BASE_URL,
        temperature=temperature,
        max_retries=1,
    )


def get_judge_llm(temperature=0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=JUDGE_MODEL_NAME,
        api_key=JUDGE_API_KEY,
        base_url=JUDGE_BASE_URL,
        temperature=temperature,
        max_retries=3,
    )


def generate_synthetic_prompt() -> str:
    room_types = ["bedroom", "living room", "kitchen", "office", "studio"]
    selected = random.choice(room_types)

    llm = get_judge_llm(temperature=0.8)

    sys = "You are a creative writer generating prompts for 3D scene synthesis."
    usr = f"Generate a description of a {selected}. Mention 3-5 objects. Max 40 words. Output raw text only."

    try:
        resp = llm.invoke([("system", sys), ("user", usr)])
        return str(resp.content).strip().replace('"', "")
    except:
        return f"A simple {selected} with a table and chair."


def judge_candidates(
    agent_name: str, system_prompt: str, user_input: str, candidates: List[str]
) -> Dict[str, Any]:

    judge_prompt = f"""
You are an expert judge evaluating AI responses for a specific agent in a 3D scene pipeline.

AGENT NAME: {agent_name}

--- SYSTEM PROMPT (Instructions given to the models) ---
{system_prompt}

--- USER INPUT ---
{user_input}

--- CANDIDATE RESPONSES ---
Candidate 0:
{candidates[0]}

Candidate 1:
{candidates[1]}

Candidate 2:
{candidates[2]}

Candidate 3:
{candidates[3]}

--- TASK ---
1. Evaluate which response best follows the System Prompt and JSON format requirements (if any).
2. JSON validity is CRITICAL for 'entity_extraction', 'relational_inference', etc.
3. Select the BEST response (most accurate, valid JSON).
4. Select the WORST response (hallucinated, invalid JSON, or missing logic).

Return JSON ONLY:
{{
  "best_index": <int 0-3>,
  "worst_index": <int 0-3>,
  "rationale": "<short explanation>"
}}
"""
    llm = get_judge_llm(temperature=0.0)

    try:
        resp = llm.invoke([("user", judge_prompt)])
        content = str(resp.content)
        data = json.loads(_extract_json_str(content))
        return data
    except Exception as e:
        print(f"  [Judge Error] {e}")
        return None


def generate_responses_safely(llm, system, user, n=1) -> List[str]:
    results = []
    for _ in range(n):
        try:
            resp = llm.invoke([("system", system), ("user", user)])
            content = str(resp.content)
            if not isinstance(content, str):
                content = json.dumps(content)
            results.append(content)
        except Exception:
            results.append("{}")
    return results


if __name__ == "__main__":
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting DPO Generation: Student ({STUDENT_MODEL_NAME}) vs Teacher ({JUDGE_MODEL_NAME})"
    )
    print(f"Output: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "a", encoding="utf8") as f_out:

        for i in tqdm(range(NUM_SAMPLES), desc="Scenes"):

            prompt_text = generate_synthetic_prompt()

            try:
                scene_result = run_multi_agents_scene(prompt_text, enrich_scene=True)
                traces = scene_result.get("traces", {})
            except Exception as e:
                print(f"  [Pipeline Error] {e}")
                continue

            for agent_name, trace_data in traces.items():

                sys_p = trace_data["system_prompt"]

                input_data = trace_data["input"]
                usr_c = ""

                if agent_name == "planner":
                    usr_c = input_data["effective_prompt"]
                elif agent_name == "entity_extraction":
                    usr_c = f"User prompt:\n{input_data['prompt']}"
                elif agent_name == "attribute_enrichment":
                    usr_c = input_data["user_prompt"]
                elif agent_name == "relational_inference":
                    usr_c = json.dumps(input_data, indent=2)
                elif agent_name == "attribute_extraction":
                    rec_input = {
                        "prompt": input_data["prompt"],
                        "nodes": input_data["nodes_before"],
                    }
                    usr_c = json.dumps(rec_input, indent=2, ensure_ascii=False)
                elif agent_name == "size_scale_estimation":
                    rec_input = {
                        "prompt": input_data["prompt"],
                        "blueprint": input_data["blueprint_before"],
                        "anchor": input_data["anchor"],
                    }
                    usr_c = json.dumps(rec_input, indent=2, ensure_ascii=False)
                elif agent_name == "graph_formalization":
                    rec_input = {"blueprint": input_data["blueprint_before"]}
                    usr_c = json.dumps(rec_input, indent=2, ensure_ascii=False)
                else:
                    continue

                candidates = []

                student_llm = get_student_llm(temperature=0.8)
                candidates.extend(
                    generate_responses_safely(student_llm, sys_p, usr_c, n=2)
                )

                actual_output_obj = trace_data["output"]
                actual_output_str = ""

                if agent_name == "planner":
                    actual_output_str = actual_output_obj["plan_notes"]
                elif agent_name == "attribute_enrichment":
                    actual_output_str = actual_output_obj["enriched_prompt"]
                elif agent_name == "entity_extraction":
                    actual_output_str = json.dumps(
                        {"nodes": actual_output_obj["nodes"]}, indent=2
                    )
                elif agent_name == "relational_inference":
                    actual_output_str = json.dumps(
                        {"edges": actual_output_obj["edges"]}, indent=2
                    )
                elif agent_name == "attribute_extraction":
                    actual_output_str = json.dumps(
                        {"nodes": actual_output_obj["nodes_after"]}, indent=2
                    )
                elif agent_name == "size_scale_estimation":
                    actual_output_str = json.dumps(
                        actual_output_obj["blueprint_after"], indent=2
                    )
                elif agent_name == "graph_formalization":
                    actual_output_str = json.dumps(
                        actual_output_obj["scene_graph"], indent=2
                    )

                candidates.append(actual_output_str)

                teacher_llm = get_judge_llm(temperature=0.7)
                candidates.extend(
                    generate_responses_safely(teacher_llm, sys_p, usr_c, n=1)
                )

                verdict = judge_candidates(agent_name, sys_p, usr_c, candidates)

                if (
                    verdict
                    and verdict.get("best_index") is not None
                    and verdict.get("worst_index") is not None
                ):
                    best_i = verdict["best_index"]
                    worst_i = verdict["worst_index"]

                    if best_i != worst_i:
                        dpo_entry = {
                            "agent": agent_name,
                            "prompt": usr_c,
                            "system": sys_p,
                            "chosen": candidates[best_i],
                            "rejected": candidates[worst_i],
                            "rationale": verdict.get("rationale", ""),
                        }

                        f_out.write(json.dumps(dpo_entry, ensure_ascii=False) + "\n")
                        f_out.flush()

            # Cleanup
            del scene_result
            del traces
            gc.collect()
            time.sleep(0.5)

    print("\nDPO Generation Complete.")
