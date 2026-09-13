from __future__ import annotations
from typing import Dict, Any, List, Set, Optional
import copy
import re
import json
from pathlib import Path


class PipelineScene:

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    @property
    def objects(self) -> List[Dict[str, Any]]:
        return self._data.get("objects", [])

    @property
    def relationships(self) -> List[Dict[str, Any]]:
        return self._data.get("relationships", [])

    def to_json(self, path: Path):
        path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf8",
        )

    def set_composition_metadata(self, meta: Dict[str, str]):
        self._data.setdefault("composition", {})
        self._data["composition"].update(meta)

    def apply_gat_predictions(
        self, gat_output: Dict[str, Any], room_dims: Dict[str, float]
    ):

        predictions = gat_output.get("predictions", [])
        preds_by_id = {p["id"]: p for p in predictions}

        room_w = room_dims.get("width_m", 4.0)
        room_h = room_dims.get("height_m", 2.5)
        room_d = room_dims.get("depth_m", 3.0)

        for obj in self.objects:
            idx = int(obj["index"])
            pred = preds_by_id.get(idx)
            if not pred:
                continue

            pred_pos_norm = pred["prediction"]["position"]
            pred_rot = pred["prediction"]["rotation_quaternion"]

            obj["center"] = [
                pred_pos_norm[0] * room_w,
                pred_pos_norm[1] * room_h,
                pred_pos_norm[2] * room_d,
            ]

            obj["rotation"] = pred_rot

            bbox = obj.get("bounding_box", [0.1, 0.1, 0.1])
            obj["bounding_box"] = [
                bbox[0] * room_w,
                bbox[1] * room_h,
                bbox[2] * room_d,
            ]

    @classmethod
    def from_scene_graph(
        cls, scene_graph: Dict[str, Any], drop_environment: bool = True
    ) -> "PipelineScene":

        sg = copy.deepcopy(scene_graph)
        nodes = sg.get("nodes", [])
        edges = sg.get("edges", [])

        if drop_environment:
            nodes = [n for n in nodes if n.get("category") != "environment"]
            kept_ids = {n["id"] for n in nodes}
            edges = [
                e
                for e in edges
                if e.get("source") in kept_ids and e.get("target") in kept_ids
            ]

        objects = []
        relationships = []
        existing_labels: Set[str] = set()
        id_to_index: Dict[str, int] = {}

        for idx, node in enumerate(nodes):
            node_id = node.get("id")
            if not node_id:
                continue

            id_to_index[node_id] = idx

            raw_label = node.get("label") or node.get("name") or "Object"
            label = cls._build_canonical_label(raw_label, existing_labels)

            attrs = node.get("attributes", {}) or {}
            dims = cls._resolve_dimensions(attrs)

            prompt = cls._build_prompt(node, attrs, raw_label)

            objects.append(
                {
                    "id": node_id,
                    "index": idx,
                    "label": label,
                    "category": node.get("category"),
                    "prompt": prompt,
                    "bounding_box": list(dims),
                    "center": [0.0, 0.0, 0.0],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                    "attributes": attrs,
                    "mesh": None,
                }
            )

        for obj in objects:
            bbox = obj["bounding_box"]
            obj["_volume"] = bbox[0] * bbox[1] * bbox[2]
        objects.sort(key=lambda o: o["_volume"], reverse=True)

        for new_idx, obj in enumerate(objects):
            obj["index"] = new_idx
            id_to_index[obj["id"]] = new_idx
            del obj["_volume"]

        if objects:
            anchor = objects[0]
            print(f"[PipelineScene] Anchor object: {anchor['label']}")

        for e in edges:
            src_id, tgt_id = e.get("source"), e.get("target")
            if src_id in id_to_index and tgt_id in id_to_index:
                relationships.append(
                    {
                        "source_index": id_to_index[src_id],
                        "target_index": id_to_index[tgt_id],
                        "source_id": src_id,
                        "target_id": tgt_id,
                        "relation": e.get("relation"),
                    }
                )

        return cls(
            {
                "objects": objects,
                "relationships": relationships,
                "metadata": sg.get("metadata", {}),
            }
        )

    @staticmethod
    def _build_canonical_label(raw_label: str, existing: Set[str]) -> str:
        if not raw_label:
            raw_label = "Object"
        label = re.sub(r"[^0-9a-zA-Z_]+", "_", raw_label.strip())
        if not label:
            label = "Object"
        if label[0].isalpha():
            label = label[0].upper() + label[1:]

        base = label
        suffix = 1
        while label in existing:
            label = f"{base}_{suffix}"
            suffix += 1
        existing.add(label)
        return label

    @staticmethod
    def _resolve_dimensions(attrs: Dict) -> tuple:
        room_scaled = attrs.get("room_scaled_dim") or {}
        dims_m = attrs.get("dimensions_m") or {}

        def _get(src, k):
            return float(src.get(k, 0.0))

        if room_scaled:
            return (
                _get(room_scaled, "width"),
                _get(room_scaled, "height"),
                _get(room_scaled, "depth"),
            )
        return (
            _get(dims_m, "width_m"),
            _get(dims_m, "height_m"),
            _get(dims_m, "depth_m"),
        )

    @staticmethod
    def _build_prompt(node, attrs, raw_label):

        def _clean_attr(val):
            if not isinstance(val, str):
                return None
            val = val.strip()

            if "|" in val:
                val = val.split("|")[0].strip()

            if val.lower() in ["null", "none", "unknown", "n/a", ""]:
                return None
            return val

        color = _clean_attr(attrs.get("color"))
        style = _clean_attr(attrs.get("style"))
        material = _clean_attr(attrs.get("material"))

        parts = []
        seen = set()

        for p in [color, style, material]:
            if p and p.lower() not in seen:
                parts.append(p)
                seen.add(p.lower())

        name = node.get("name") or raw_label
        if name and name.strip():

            name_lower = name.lower()
            if not any(p.lower() in name_lower for p in parts):
                parts.append(name.strip())
            else:
                parts.append(name.strip())

        final_prompt = " ".join(parts)
        return final_prompt if final_prompt else "Object"
