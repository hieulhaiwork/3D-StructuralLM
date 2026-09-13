# src/complete_pipeline/nodes/gat_node.py

import json
import subprocess
from pathlib import Path
from ..core.base import PipelineNode, RunContext
from ..core.scene_data import PipelineScene
from .. import config


class GATNode(PipelineNode):
    def __init__(
        self,
        checkpoint_rel: str = "checkpoints/model_phase1234.pt",
        config_rel: str = "configs/configs_syn.yaml",
    ):
        super().__init__("graph_attention_network")
        self.checkpoint_rel = checkpoint_rel
        self.config_rel = config_rel

    def process(self, context: RunContext, scene: PipelineScene) -> PipelineScene:
        if scene is None:
            raise ValueError("GATNode requires a valid pipeline scene input.")

        node_dir = context.get_node_dir(self.name)
        gat_root = config.GAT_ROOT

        scene_json_path = node_dir / "scene_for_gat.json"
        gat_input = self._build_gat_input(scene)
        scene_json_path.write_text(json.dumps(gat_input, indent=2), encoding="utf8")
        print(f"[{self.name}] Wrote scene_for_gat.json -> {scene_json_path}")

        output_json_path = node_dir / "gat_output.json"
        cmd = [
            "python",
            "inference.py",
            "--scene",
            str(scene_json_path),
            "--checkpoint",
            str(gat_root / self.checkpoint_rel),
            "--config",
            str(gat_root / self.config_rel),
            "--output",
            str(output_json_path),
        ]
        self._run_cmd(cmd, cwd=gat_root)

        gat_output = json.loads(output_json_path.read_text(encoding="utf8"))
        room_dims = scene.data.get("room_dimensions_m", {})
        scene.apply_gat_predictions(gat_output, room_dims)

        scene.to_json(node_dir / "pipeline_scene_after_gat.json")
        return scene

    def _build_gat_input(self, scene: PipelineScene) -> dict:
        objects = []
        for obj in scene.objects:
            objects.append(
                {
                    "id": int(obj["index"]),
                    "name": f"{obj['label']}_{obj['index']}",
                    "label": obj["label"],
                    "normalized_bounding_box": obj["bounding_box"],
                    "normalized_relative_center": obj.get("center", [0.0, 0.0, 0.0]),
                    "rot": obj.get("rotation", [1.0, 0.0, 0.0, 0.0]),
                }
            )

        relationships = []
        for rel in scene.relationships:
            relationships.append(
                {
                    "obj_id1": int(rel["source_index"]),
                    "obj_id2": int(rel["target_index"]),
                    "relation": rel["relation"],
                }
            )

        return {"objects": objects, "relationships": relationships}

    def _run_cmd(self, cmd, cwd):
        print("\n[GAT] Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(cwd))
