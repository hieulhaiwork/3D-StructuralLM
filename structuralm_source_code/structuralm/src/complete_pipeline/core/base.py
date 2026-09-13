from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from .scene_data import PipelineScene


@dataclass
class RunContext:
    run_id: str
    run_dir: Path
    gpu_id: int = 0
    test_mode: bool = False

    def get_node_dir(self, node_name: str) -> Path:
        path = self.run_dir / node_name
        path.mkdir(parents=True, exist_ok=True)
        return path


class PipelineNode(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(
        self, context: RunContext, scene: PipelineScene | None
    ) -> PipelineScene:
        pass
