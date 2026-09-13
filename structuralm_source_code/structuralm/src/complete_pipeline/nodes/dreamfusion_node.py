import subprocess
from pathlib import Path
from typing import Optional, Dict
from ..core.base import PipelineNode, RunContext
from ..core.scene_data import PipelineScene
from .. import config


class DreamFusionNode(PipelineNode):
    def __init__(
        self,
        use_existing_runs: bool = False,
        existing_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__("dreamfusion")
        self.use_existing_runs = use_existing_runs
        self.existing_map = existing_map
        self.df_root = config.DREAMFUSION_ROOT
        self.runs_root = self.df_root / "outputs" / "dreamfusion-if"

    def process(self, context: RunContext, scene: PipelineScene) -> PipelineScene:
        node_dir = context.get_node_dir(self.name)
        print(f"[{self.name}] Runs root: {self.runs_root}")

        force_existing = context.test_mode or self.use_existing_runs

        for obj in scene.objects:
            label = obj["label"]
            prompt = obj["prompt"]
            print(f"\n[{self.name}] Object {obj['index']} label={label}")

            if force_existing:
                trained_dir = self._find_existing_run(label, prompt)
            else:
                trained_dir = self._train_new(prompt, context.gpu_id)

            mesh_data = self._export_mesh(trained_dir, label, node_dir, context.gpu_id)
            obj["mesh"] = mesh_data

        return scene

    def _find_existing_run(self, label: str, prompt: str) -> Path:
        if self.existing_map and label in self.existing_map:
            path = self.runs_root / self.existing_map[label]
            if path.is_dir():
                return path

        slug = self._slugify(prompt)
        candidates = [
            p
            for p in self.runs_root.iterdir()
            if p.is_dir() and p.name.startswith(slug + "@")
        ]
        if not candidates:
            raise FileNotFoundError(f"No runs found for slug '{slug}'")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _train_new(self, prompt: str, gpu: int) -> Path:
        cmd = [
            "python",
            "launch.py",
            "--config",
            "configs/dreamfusion-if.yaml",
            "--train",
            "--gpu",
            str(gpu),
            f'system.prompt_processor.prompt="{prompt}"',
            "system.background.random_aug=true",
        ]
        self._run_cmd(cmd, cwd=self.df_root)

        candidates = [p for p in self.runs_root.glob("*@*") if p.is_dir()]
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _export_mesh(self, run_dir: Path, label: str, node_dir: Path, gpu: int) -> dict:
        parsed_config = run_dir / "configs" / "parsed.yaml"
        resume_ckpt = run_dir / "ckpts" / "last.ckpt"

        cmd = [
            "python",
            "launch.py",
            "--config",
            str(parsed_config),
            "--export",
            "--gpu",
            str(gpu),
            f"resume={resume_ckpt}",
            "system.exporter_type=mesh-exporter",
            "system.exporter.context_type=cuda",
            "system.exporter.fmt=obj-mtl",
        ]
        self._run_cmd(cmd, cwd=self.df_root)

        save_dir = run_dir / "save"
        export_dirs = [
            p for p in save_dir.iterdir() if p.is_dir() and p.name.endswith("-export")
        ]
        latest = max(export_dirs, key=lambda p: p.stat().st_mtime)

        src_obj, src_mtl = latest / "model.obj", latest / "model.mtl"
        dst_obj, dst_mtl = node_dir / f"{label}.obj", node_dir / f"{label}.mtl"

        dst_obj.write_bytes(src_obj.read_bytes())
        dst_mtl.write_bytes(src_mtl.read_bytes())
        self._patch_mtllib(dst_obj, dst_mtl.name)

        return {"obj": str(dst_obj), "mtl": str(dst_mtl)}

    def _slugify(self, text: str) -> str:
        s = text.strip().lower()
        for ch in [" ", "/", "\\", ":", "?", "#", "@"]:
            s = s.replace(ch, "_")
        while "__" in s:
            s = s.replace("__", "_")
        return s

    def _patch_mtllib(self, obj_path: Path, new_mtl: str):
        lines = obj_path.read_text().splitlines()
        lines = [f"mtllib {new_mtl}" if l.startswith("mtllib ") else l for l in lines]
        if not any(l.startswith("mtllib") for l in lines):
            lines.insert(0, f"mtllib {new_mtl}")
        obj_path.write_text("\n".join(lines) + "\n")

    def _run_cmd(self, cmd, cwd):
        print(f"\n[DreamFusion] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
