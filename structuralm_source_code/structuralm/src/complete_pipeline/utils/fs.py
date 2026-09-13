from __future__ import annotations
from pathlib import Path
from datetime import datetime
from .. import config


def new_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_run_dir(run_id: str) -> Path:
    run_dir = config.CACHE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_node_dir(run_dir: Path, node_name: str) -> Path:
    node_dir = run_dir / node_name
    node_dir.mkdir(parents=True, exist_ok=True)
    return node_dir
