import json
import sys
from pathlib import Path
from ..core.base import PipelineNode, RunContext
from ..core.scene_data import PipelineScene
from .. import config


class CompositionsNode(PipelineNode):
    def __init__(
        self,
        scene_id: str = "scene_001",
        include_floor: bool = True,
        y_rotation_degrees: float = 0.0,
    ):
        super().__init__("compositions")
        self.scene_id = scene_id
        self.include_floor = include_floor
        self.y_rotation = y_rotation_degrees

    def process(self, context: RunContext, scene: PipelineScene) -> PipelineScene:
        node_dir = context.get_node_dir(self.name)
        df_node_dir = context.get_node_dir("dreamfusion")

        pos_data = {self.scene_id: []}
        for obj in scene.objects:
            pos_data[self.scene_id].append(
                {
                    "label": obj["label"],
                    "bounding_box": obj["bounding_box"],
                    "rotation": obj.get("rotation"),
                    "center": obj.get("center"),
                }
            )

        pos_json_path = node_dir / "pos.json"
        pos_json_path.write_text(json.dumps(pos_data, indent=2), encoding="utf8")

        sys.path.append(str(config.COMPOSITIONS_ROOT))
        from Composition.composition import SceneOrchestrator

        print(f"[{self.name}] Initializing Orchestrator with y_rot={self.y_rotation}")
        orchestrator = SceneOrchestrator(y_rotation_degrees=self.y_rotation)

        rel_strings = []
        for rel in scene.relationships:
            src = scene.objects[rel["source_index"]]["label"]
            tgt = scene.objects[rel["target_index"]]["label"]
            rel_strings.append(
                {"source": src, "target": tgt, "relation": rel["relation"]}
            )

        orchestrator.load_scene_from_json(
            json_path=str(pos_json_path),
            obj_folder=str(df_node_dir),
            relationships=rel_strings,
        )

        output_obj = node_dir / f"{self.scene_id}.obj"
        orchestrator.export_scene(self.scene_id, str(output_obj), self.include_floor)

        scene.set_composition_metadata(
            {
                "scene_id": self.scene_id,
                "output_obj": str(output_obj),
                "pos_json": str(pos_json_path),
            }
        )

        scene.to_json(node_dir / "pipeline_scene_final.json")
        return scene
