import os
import json
import re
import copy
import yaml
from typing import Dict, Any
from typing_extensions import TypedDict
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_FILE = SCRIPT_DIR / "agents.yaml"


def load_prompts() -> Dict[str, str]:
    with open(PROMPT_FILE, "r", encoding="utf8") as f:
        return yaml.safe_load(f)


PROMPTS: Dict[str, str] = load_prompts()


def make_model() -> ChatOpenAI:
    api_key = os.getenv("MY_LLM_API_KEY")
    base_url = os.getenv("MY_LLM_BASE_URL")
    model_name = os.getenv("SCENE_MODEL_NAME", "Llama-3.3-70B-Instruct")

    missing = [
        name
        for name, val in [
            ("MY_LLM_API_KEY", api_key),
            ("MY_LLM_BASE_URL", base_url),
        ]
        if not val
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variables for multi-agent LLM:\n"
            f"  {', '.join(missing)}\n"
            "Set them before running (e.g. in run_pipeline.sh)."
        )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
    )


_llm = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = make_model()
    return _llm


def _extract_json_str(raw: str) -> str:

    if not isinstance(raw, str):
        raw = str(raw)

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fenced:
        candidate = fenced[0].strip()
        if candidate.startswith("{") or candidate.startswith("["):
            return candidate

    stack = []
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if start is None:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack:
                    return raw[start : i + 1]

    raise ValueError(f"Could not extract JSON from LLM response:\n{raw}")


def llm_json_call(system_prompt: str, user_content: str) -> Any:

    llm = get_llm()
    resp = llm.invoke([("system", system_prompt), ("user", user_content)])

    content = resp.content
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)

    try:
        json_str = _extract_json_str(content)
        return json.loads(json_str)
    except Exception as e:
        raise ValueError(
            "\n❌ LLM JSON PARSE ERROR\n"
            "---------------------------------------\n"
            f"System prompt:\n{system_prompt}\n\n"
            f"User content:\n{user_content}\n\n"
            f"Raw LLM output:\n{content}\n\n"
        ) from e


class SceneState(TypedDict, total=False):
    user_prompt: str
    enriched_prompt: str
    blueprint: Dict[str, Any]
    plan_notes: str
    scene_graph: Dict[str, Any]
    traces: Dict[str, Any]
    no_obj_enrich: bool


OBJECT_DIMENSIONS_DB: Dict[str, Dict[str, float]] = {}


def database_lookup(object_name: str) -> Dict[str, float]:
    name = object_name.lower()
    for key, dims in OBJECT_DIMENSIONS_DB.items():
        if key in name:
            return dims
    return {}


def choose_anchor_object(nodes: list[dict]) -> str | None:
    best = None
    best_score = -1

    for n in nodes:
        dims = n.get("attributes", {}).get("dimensions_m", {})
        w = dims.get("width_m", 1.0)
        h = dims.get("height_m", 1.0)
        d = dims.get("depth_m", 1.0)
        score = w * h * d
        if score > best_score:
            best = n["id"]
            best_score = score

    if best:
        return best

    for n in nodes:
        if n["category"] == "environment":
            return n["id"]

    return nodes[0]["id"] if nodes else None


def planning_agent(state: SceneState) -> SceneState:
    print("\n\n🔵 [PLANNER AGENT] Thinking...")

    traces = state.get("traces", {})
    user_prompt = state["user_prompt"]
    effective_prompt = state.get("enriched_prompt") or user_prompt
    system_prompt = PROMPTS["planner"]

    resp = get_llm().invoke([("system", system_prompt), ("user", effective_prompt)])
    plan_text = resp.content if isinstance(resp.content, str) else str(resp.content)
    plan_text = plan_text.strip()

    blueprint = state.get("blueprint") or {"nodes": [], "edges": [], "metadata": {}}
    metadata = blueprint.setdefault("metadata", {})
    metadata.setdefault("original_prompt", user_prompt)
    if "enriched_prompt" not in metadata and "enriched_prompt" in state:
        metadata["enriched_prompt"] = state["enriched_prompt"]
    metadata["plan"] = plan_text

    traces["planner"] = {
        "system_prompt": system_prompt,
        "input": {
            "user_prompt": user_prompt,
            "effective_prompt": effective_prompt,
        },
        "output": {
            "plan_notes": plan_text,
            "blueprint_after": copy.deepcopy(blueprint),
        },
    }

    print(f"📝 **Generated Plan:**\n{plan_text}\n")

    return {"plan_notes": plan_text, "blueprint": blueprint, "traces": traces}


def entity_extraction_agent(state: SceneState) -> SceneState:
    print("\n\n🔵 [ENTITY AGENT] Identifying objects...")

    prompt = state.get("enriched_prompt") or state["user_prompt"]
    traces = state.get("traces", {})
    blueprint = state["blueprint"]
    system_prompt = PROMPTS["entity_extraction"]
    user_content = f"User prompt:\n{prompt}"

    data = llm_json_call(system_prompt, user_content)
    blueprint["nodes"] = data.get("nodes", [])

    print("📦 **Extracted Entities:**")
    for n in blueprint["nodes"]:
        print(f"- {n.get('name')} (Label: {n.get('label')})")

    traces["entity_extraction"] = {
        "system_prompt": system_prompt,
        "input": {
            "prompt": prompt,
        },
        "output": {
            "nodes": copy.deepcopy(blueprint["nodes"]),
            "blueprint_after": copy.deepcopy(blueprint),
        },
    }
    return {"blueprint": blueprint, "traces": traces}


def relational_inference_agent(state: SceneState) -> SceneState:
    print("\n\n🔵 [RELATION AGENT] Connecting objects...")

    prompt = state.get("enriched_prompt") or state["user_prompt"]
    traces = state.get("traces", {})
    blueprint = state["blueprint"]
    nodes = blueprint["nodes"]
    system_prompt = PROMPTS["relational_inference"]
    user_content = json.dumps({"prompt": prompt, "nodes": nodes}, indent=2)

    data = llm_json_call(system_prompt, user_content)
    blueprint["edges"] = data.get("edges", [])

    print("🔗 **Relationships Found:**")
    if not blueprint["edges"]:
        print("- None detected.")
    for e in blueprint["edges"]:
        print(f"- {e['source']} -> {e['relation']} -> {e['target']}")

    traces["relational_inference"] = {
        "system_prompt": system_prompt,
        "input": {
            "prompt": prompt,
            "nodes": copy.deepcopy(nodes),
        },
        "output": {
            "edges": copy.deepcopy(blueprint["edges"]),
            "blueprint_after": copy.deepcopy(blueprint),
        },
    }
    return {"blueprint": blueprint, "traces": traces}


def attribute_enrichment_agent(state: SceneState) -> SceneState:
    print("\n\n🔵 [ENRICHMENT AGENT] Enhancing description...")

    user_prompt = state.get("enriched_prompt") or state["user_prompt"]
    traces = state.get("traces", {})
    no_obj_enrich = state.get("no_obj_enrich", False)
    system_prompt = PROMPTS["attribute_enrichment"]

    if no_obj_enrich:
        system_prompt += (
            "\n\nSTRICT CONSTRAINT: Do NOT add any new objects or entities "
            "that were not mentioned in the original user prompt. "
            "You may only enrich the existing objects with descriptions "
            "of their color, material, style, size, or texture."
        )

    resp = get_llm().invoke([("system", system_prompt), ("user", user_prompt)])
    enriched = resp.content if isinstance(resp.content, str) else str(resp.content)
    enriched = resp.content.strip()

    print(f"✨ **Enriched Description:**\n{enriched}\n")

    blueprint = state.get("blueprint") or {"nodes": [], "edges": [], "metadata": {}}
    metadata = blueprint.setdefault("metadata", {})
    metadata["enriched_prompt"] = enriched

    traces["attribute_enrichment"] = {
        "system_prompt": system_prompt,
        "input": {
            "user_prompt": user_prompt,
        },
        "output": {
            "enriched_prompt": enriched,
            "blueprint_after": copy.deepcopy(blueprint),
        },
    }
    return {"enriched_prompt": enriched, "blueprint": blueprint, "traces": traces}


def attribute_extraction_agent(state: SceneState) -> SceneState:
    print("\n\n🔵 [ATTRIBUTE AGENT] detailing objects...")
    prompt = state.get("enriched_prompt") or state["user_prompt"]
    traces = state.get("traces", {})
    blueprint = state["blueprint"]
    nodes = blueprint.get("nodes", [])
    system_prompt = PROMPTS["attribute_extraction"]
    user_content = json.dumps(
        {"prompt": prompt, "nodes": nodes}, indent=2, ensure_ascii=False
    )

    data = llm_json_call(system_prompt, user_content)

    attr_by_id = {n["id"]: n.get("attributes", {}) for n in data.get("nodes", [])}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue

        node.setdefault("attributes", {})

        extracted = attr_by_id.get(node_id, {})
        for key in ("color", "material", "style"):
            value = extracted.get(key, None)
            if value is not None:
                node["attributes"][key] = value

    blueprint["nodes"] = nodes
    traces["attribute_extraction"] = {
        "system_prompt": system_prompt,
        "input": {
            "prompt": prompt,
            "nodes_before": copy.deepcopy(nodes),
        },
        "output": {
            "nodes_after": copy.deepcopy(nodes),
            "blueprint_after": copy.deepcopy(blueprint),
        },
    }
    return {"blueprint": blueprint, "traces": traces}


def size_scale_estimation_agent(state: SceneState) -> SceneState:
    print("\n\n🔵 [SCALE AGENT] Calculating dimensions...")
    prompt = state.get("enriched_prompt") or state["user_prompt"]
    traces = state.get("traces", {})
    blueprint = state["blueprint"]
    nodes = blueprint["nodes"]

    for node in nodes:
        obj = node.get("name") or node["id"]
        dims_from_db = database_lookup(obj)
        if dims_from_db:
            node.setdefault("attributes", {})
            existing_dims = node["attributes"].get("dimensions_m", {})
            merged_dims = {**dims_from_db, **existing_dims}
            node["attributes"]["dimensions_m"] = merged_dims

    for node in nodes:
        node.setdefault("attributes", {})
        dims = node["attributes"].get("dimensions_m", {})
        dims.setdefault("width_m", 1.0)
        dims.setdefault("height_m", 1.0)
        dims.setdefault("depth_m", 1.0)
        node["attributes"]["dimensions_m"] = dims

    anchor = choose_anchor_object(nodes)
    system_prompt = f"""
You are the Size & Scale Estimation Agent.

Your tasks:

1. For EVERY node in blueprint["nodes"], ensure:
   node["attributes"]["dimensions_m"] = {{
     "width_m":  <float, meters>,
     "height_m": <float, meters>,
     "depth_m":  <float, meters>
   }}

2. Use the ANCHOR OBJECT as a key reference for relative size
   (e.g. if it's a sofa, other furniture must be sized consistently):

   ANCHOR OBJECT ID: "{anchor}"

3. Estimate the OVERALL ROOM size and write it into:
   blueprint["metadata"]["room_dimensions_m"] = {{
     "width_m":  <float, meters>,  # horizontal extent (left-right)
     "height_m": <float, meters>,  # floor to ceiling
     "depth_m":  <float, meters>   # front-back
   }}

4. Dimensions must be physically plausible:
   - Humans typically ~1.6–2.0m tall.
   - Sofas typically ~1.8–2.5m wide.
   - Coffee tables typically ~0.8–1.2m long.
   - Rooms typically 2.4–3.0m tall, 2.5–6.0m wide, etc.

5. Keep all nodes, edges, and existing metadata fields.
   Only adjust numeric dimension values to make the scene realistic.

You MUST return the FULL blueprint JSON with this structure:

{{
  "nodes": [...],
  "edges": [...],
  "metadata": {{ ... }}
}}

No comments or extra text.
""".strip()

    uc = json.dumps(
        {"prompt": prompt, "blueprint": blueprint, "anchor": anchor},
        indent=2,
        ensure_ascii=False,
    )
    sized = llm_json_call(system_prompt, uc)

    meta = sized.get("metadata", {}).get("room_dimensions_m", {})
    print(f"📏 **Room Size:** {meta.get('width_m')}m x {meta.get('depth_m')}m")

    traces["size_scale_estimation"] = {
        "system_prompt": system_prompt,
        "input": {
            "prompt": prompt,
            "blueprint_before": copy.deepcopy(blueprint),
            "anchor": anchor,
        },
        "output": {
            "blueprint_after": copy.deepcopy(sized),
        },
    }
    return {"blueprint": sized, "traces": traces}


def graph_formalization_agent(state: SceneState) -> SceneState:
    print("\n\n🔵 [FINALIZING] Building graph...")

    traces = state.get("traces", {})
    blueprint = state["blueprint"]
    system_prompt = PROMPTS["graph_formalization"]
    user_content = json.dumps({"blueprint": blueprint}, indent=2, ensure_ascii=False)

    scene_graph = llm_json_call(system_prompt, user_content)
    scene_graph = add_room_scaled_dimensions(scene_graph)

    print("✅ **Graph Complete.**")

    traces["graph_formalization"] = {
        "system_prompt": system_prompt,
        "input": {
            "blueprint_before": copy.deepcopy(blueprint),
        },
        "output": {
            "scene_graph": copy.deepcopy(scene_graph),
        },
    }
    return {"blueprint": scene_graph, "scene_graph": scene_graph, "traces": traces}


def add_room_scaled_dimensions(scene_graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add per-node 'room_scaled_dim' attributes based on dimensions_m and room_dimensions_m.

    room_scaled_dim = {
        "width":  width_m  / room_width_m,
        "height": height_m / room_height_m,
        "depth":  depth_m  / room_depth_m
    }
    """

    metadata = scene_graph.get("metadata", {})
    room_dims = metadata.get("room_dimensions_m")

    if not room_dims:
        max_w = max_h = max_d = 0.0
        for node in scene_graph.get("nodes", []):
            attrs = node.get("attributes", {})
            dims = attrs.get("dimensions_m", {})
            try:
                max_w = max(max_w, float(dims.get("width_m", 0.0)))
                max_h = max(max_h, float(dims.get("height_m", 0.0)))
                max_d = max(max_d, float(dims.get("depth_m", 0.0)))
            except (TypeError, ValueError):
                continue

        room_dims = {
            "width_m": max_w or 1.0,
            "height_m": max_h or 1.0,
            "depth_m": max_d or 1.0,
        }
        metadata["room_dimensions_m"] = room_dims
        scene_graph["metadata"] = metadata

    rw = float(room_dims.get("width_m", 1.0) or 1.0)
    rh = float(room_dims.get("height_m", 1.0) or 1.0)
    rd = float(room_dims.get("depth_m", 1.0) or 1.0)

    for node in scene_graph.get("nodes", []):
        attrs = node.setdefault("attributes", {})
        dims = attrs.get("dimensions_m")
        if not dims:
            continue

        try:
            w = float(dims.get("width_m", 0.0))
            h = float(dims.get("height_m", 0.0))
            d = float(dims.get("depth_m", 0.0))
        except (TypeError, ValueError):
            continue

        attrs["room_scaled_dim"] = {
            "width": w / rw if rw else None,
            "height": h / rh if rh else None,
            "depth": d / rd if rd else None,
        }

    return scene_graph


def build_scene_graph_workflow(enrich_scene: bool = True):

    graph = StateGraph(SceneState)

    graph.add_node("planner", planning_agent)
    graph.add_node("entity_extraction", entity_extraction_agent)
    graph.add_node("attribute_extraction", attribute_extraction_agent)
    graph.add_node("relational_inference", relational_inference_agent)
    graph.add_node("size_scale_estimation", size_scale_estimation_agent)
    graph.add_node("graph_formalization", graph_formalization_agent)

    if enrich_scene:
        graph.add_node("attribute_enrichment", attribute_enrichment_agent)
        graph.add_edge(START, "attribute_enrichment")
        graph.add_edge("attribute_enrichment", "planner")
    else:
        graph.add_edge(START, "planner")

    graph.add_edge("planner", "entity_extraction")
    graph.add_edge("entity_extraction", "attribute_extraction")
    graph.add_edge("attribute_extraction", "relational_inference")
    graph.add_edge("relational_inference", "size_scale_estimation")
    graph.add_edge("size_scale_estimation", "graph_formalization")
    graph.add_edge("graph_formalization", END)

    return graph.compile()


def save_graph_png(workflow, out_path: Path | None = None):
    if out_path is None:
        script_dir = Path(__file__).resolve().parent
        out_path = script_dir / "multi_agent_graph3.png"

    png_bytes = workflow.get_graph().draw_mermaid_png()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(png_bytes)

    print(f"Saved graph to {out_path}")


def run_multi_agents_scene(
    user_prompt: str, enrich_scene: bool = True, no_obj_enrich: bool = False
) -> SceneState:
    workflow = build_scene_graph_workflow(enrich_scene=enrich_scene)
    return workflow.invoke(
        {"user_prompt": user_prompt}, {"no_obj_enrich": no_obj_enrich}
    )


if __name__ == "__main__":
    workflow = build_scene_graph_workflow()
    save_graph_png(workflow)

    prompt = (
        "Create a small cozy bedroom with warm lighting, a soft duvet, "
        "a single window with curtains, a small desk, and simple wooden furniture."
    )

    prompt_astronaut = (
        "An astronaut sitting on a red sofa in a cozy living room,"
        "with a small coffee table in front of the sofa."
    )

    result = workflow.invoke({"user_prompt": prompt_astronaut})

    print("\n=== PLAN NOTES ===")
    print(result["plan_notes"])

    print("\n=== FINAL SCENE GRAPH ===")
    print(json.dumps(result["scene_graph"], indent=2, ensure_ascii=False))

    traces_out_path = Path(r"multi_agent/agent_traces_astronaut3.json")
    traces_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traces_out_path, "w", encoding="utf8") as f:
        json.dump(result.get("traces", {}), f, indent=2, ensure_ascii=False)

    print(f"\nSaved agent traces to {traces_out_path}")
