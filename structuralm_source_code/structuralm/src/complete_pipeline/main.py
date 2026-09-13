import argparse
from datetime import datetime
from pathlib import Path

from . import config
from .core.base import RunContext, PipelineNode
from .core.scene_data import PipelineScene

from .nodes.multi_agents_node import MultiAgentsNode
from .nodes.gat_node import GATNode
from .nodes.dreamfusion_node import DreamFusionNode
from .nodes.compositions_node import CompositionsNode
import warnings

warnings.filterwarnings("ignore")


class SceneGenerationPipeline:
    def __init__(self, args: argparse.Namespace):
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = config.CACHE_ROOT / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.context = RunContext(
            run_id=self.run_id, run_dir=self.run_dir, test_mode=args.test, gpu_id=0
        )
        self.nodes: list[PipelineNode] = []
        self._print_header(args)

    def add_node(self, node: PipelineNode):
        self.nodes.append(node)

    def execute(self):
        current_scene: PipelineScene = None

        for node in self.nodes:
            print(f"\n{'='*10} Executing Node: {node.name} {'='*10}")
            current_scene = node.process(self.context, current_scene)

        print(f"\n[PIPELINE] Finished. Result at: {self.run_dir}")

    def _print_header(self, args):
        print(f"[PIPELINE] Run ID: {self.run_id}")
        print(f"[PIPELINE] Test Mode: {args.test}")
        print(f"[PIPELINE] Dir: {self.run_dir}")


def _load_prompt(args):
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return Path(args.prompt_file).read_text().strip()
    return Path("prompt.txt").read_text().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt-file", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--no-enrich", action="store_true")
    parser.add_argument("--no-obj-enrich", action="store_true")
    parser.add_argument("--y-rotation", type=float, default=0.0)
    args = parser.parse_args()

    pipeline = SceneGenerationPipeline(args)
    prompt = _load_prompt(args)

    pipeline.add_node(
        MultiAgentsNode(user_prompt=prompt, enrich_scene=not args.no_enrich)
    )
    pipeline.add_node(GATNode())
    pipeline.add_node(DreamFusionNode())
    pipeline.add_node(CompositionsNode(y_rotation_degrees=args.y_rotation))

    pipeline.execute()


if __name__ == "__main__":
    main()
