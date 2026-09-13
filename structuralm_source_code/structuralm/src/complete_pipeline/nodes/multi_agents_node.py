# src/complete_pipeline/nodes/multi_agents_node.py

import json
from ..core.base import PipelineNode, RunContext
from ..core.scene_data import PipelineScene
from Multi_Agents.multi_agents import run_multi_agents_scene


class MultiAgentsNode(PipelineNode):
    def __init__(
        self, user_prompt: str, enrich_scene: bool = True, no_obj_enrich: bool = False
    ):
        super().__init__("multi_agents")
        self.user_prompt = user_prompt
        self.enrich_scene = enrich_scene
        self.no_obj_enrich = no_obj_enrich

    def process(
        self, context: RunContext, scene: PipelineScene | None
    ) -> PipelineScene:
        node_dir = context.get_node_dir(self.name)
        print(
            f"[{self.name}] Running multi-agents with prompt: {self.user_prompt[:50]}..."
        )

        result = run_multi_agents_scene(
            self.user_prompt, enrich_scene=self.enrich_scene
        )
        scene_graph = result["scene_graph"]

        pipeline_scene = PipelineScene.from_scene_graph(
            scene_graph, drop_environment=True
        )

        (node_dir / "scene_graph.json").write_text(
            json.dumps(scene_graph, indent=2, ensure_ascii=False), encoding="utf8"
        )
        pipeline_scene.to_json(node_dir / "pipeline_scene.json")
        (node_dir / "traces.json").write_text(
            json.dumps(result.get("traces", {}), indent=2, ensure_ascii=False),
            encoding="utf8",
        )

        return pipeline_scene
