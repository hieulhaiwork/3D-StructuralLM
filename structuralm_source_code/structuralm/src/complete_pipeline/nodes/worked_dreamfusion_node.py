from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import re

from .. import config
from ..utils.fs import get_node_dir

NODE_NAME = "dreamfusion"


def prompt_to_slug(prompt: str) -> str:
    slug = prompt.strip().lower()
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", slug)
    slug = slug.strip("_")
    return slug


def run_cmd(cmd, cwd: Path | None = None):
    print("\n[DreamFusion] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def patch_obj_mtllib(obj_path: Path, new_mtl_name: str):
    text = obj_path.read_text()
    lines = text.splitlines()
    new_lines = []
    replaced = False

    for line in lines:
        if line.startswith("mtllib ") and not replaced:
            new_lines.append(f"mtllib {new_mtl_name}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        new_lines.insert(0, f"mtllib {new_mtl_name}")

    obj_path.write_text("\n".join(new_lines) + "\n")


def find_latest_export_dir(run_dir: Path) -> Path:
    save_dir = run_dir / "save"
    export_dirs = [
        p for p in save_dir.iterdir() if p.is_dir() and p.name.endswith("-export")
    ]
    if not export_dirs:
        raise FileNotFoundError(f"No '*-export' dirs in {save_dir}")
    return max(export_dirs, key=lambda p: p.stat().st_mtime)


def run_dreamfusion_batch(
    run_dir: Path,
    prompts: List[str],
    gpu: int = 0,
    use_existing_runs: bool = False,
    existing_run_dirs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:

    node_dir = get_node_dir(run_dir, NODE_NAME)
    results: List[Dict[str, Any]] = []

    runs_root = config.DREAMFUSION_ROOT / "outputs" / "dreamfusion-if"

    for idx, prompt in enumerate(prompts):
        slug = prompt_to_slug(prompt)
        print(f"\n[DreamFusion] Prompt #{idx}: '{prompt}' (slug={slug})")

        if existing_run_dirs is not None and idx < len(existing_run_dirs):
            run_folder_name = existing_run_dirs[idx]
            trained_run_dir = runs_root / run_folder_name
        else:
            candidates = [
                p
                for p in runs_root.iterdir()
                if p.is_dir() and p.name.startswith(slug + "@")
            ]
            if not candidates:
                raise FileNotFoundError(f"No run dir for slug '{slug}' in {runs_root}")
            trained_run_dir = max(candidates, key=lambda p: p.stat().st_mtime)

        print(f"[DreamFusion] Using trained run dir: {trained_run_dir}")

        parsed_config = trained_run_dir / "configs" / "parsed.yaml"
        resume_ckpt = trained_run_dir / "ckpts" / "last.ckpt"

        export_cmd = [
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
        run_cmd(export_cmd, cwd=config.DREAMFUSION_ROOT)

        export_dir = find_latest_export_dir(trained_run_dir)
        src_obj = export_dir / "model.obj"
        src_mtl = export_dir / "model.mtl"

        out_base = f"{slug}_{idx}"
        dst_obj = node_dir / f"{out_base}.obj"
        dst_mtl = node_dir / f"{out_base}.mtl"

        dst_obj.write_bytes(src_obj.read_bytes())
        dst_mtl.write_bytes(src_mtl.read_bytes())
        patch_obj_mtllib(dst_obj, dst_mtl.name)

        results.append(
            {
                "prompt": prompt,
                "index": idx,
                "slug": slug,
                "trained_run_dir": trained_run_dir,
                "export_dir": export_dir,
                "obj_path": dst_obj,
                "mtl_path": dst_mtl,
            }
        )

        print(f"[DreamFusion] Cached OBJ: {dst_obj}")
        print(f"[DreamFusion] Cached MTL: {dst_mtl}")

    return results
