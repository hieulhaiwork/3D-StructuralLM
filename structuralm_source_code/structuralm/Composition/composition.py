import numpy as np
import json
import shutil  # Moved to top level
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import quaternion
import trimesh


class SceneOrchestrator:
    def __init__(
        self,
        floor_size: Tuple[float, float] = (5.0, 5.0),
        y_rotation_degrees: float = 90.0,
    ):

        self.floor_size = floor_size
        self.scenes = {}
        self.scene_floor_heights = {}
        self.scene_materials = {}
        self.scene_textures = {}

        y_rad = np.radians(y_rotation_degrees)
        self.y_rotation = np.quaternion(
            np.cos(y_rad / 2), 0.0, np.sin(y_rad / 2), 0.0
        ).normalized()

    def load_obj_with_mtl(self, obj_path: str, object_label: str) -> Dict:

        path = Path(obj_path)
        mtl_path = path.with_suffix(".mtl")

        materials_content = []
        material_mapping = {}
        texture_files = []

        if mtl_path.exists():
            with open(mtl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("newmtl "):
                        old_name = line.split(maxsplit=1)[1]
                        new_name = f"{object_label}_{old_name}"
                        material_mapping[old_name] = new_name
                        materials_content.append(f"newmtl {new_name}")
                    elif line.startswith(
                        ("map_Kd", "map_Ks", "map_bump", "bump", "map_d")
                    ):
                        parts = line.split()
                        if len(parts) >= 2:
                            texture_filename = parts[-1]
                            texture_files.append(texture_filename)
                        materials_content.append(line)
                    else:
                        materials_content.append(line)
        else:
            print(f"Warning: No .mtl file found for {obj_path}")

        vertices = []
        normals = []
        uvs = []
        faces = []

        current_mat = None

        with open(obj_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("vn "):
                    parts = line.split()
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("vt "):
                    parts = line.split()
                    uvs.append([float(parts[1]), float(parts[2])])
                elif line.startswith("usemtl "):
                    raw_mat = line.split()[1]
                    current_mat = material_mapping.get(
                        raw_mat, f"{object_label}_{raw_mat}"
                    )
                elif line.startswith("f "):
                    parts = line.split()[1:]
                    face_indices = []
                    for p in parts:
                        vals = p.split("/")
                        v_idx = int(vals[0]) - 1
                        vt_idx = int(vals[1]) - 1 if len(vals) > 1 and vals[1] else None
                        vn_idx = int(vals[2]) - 1 if len(vals) > 2 and vals[2] else None
                        face_indices.append((v_idx, vt_idx, vn_idx))

                    faces.append({"material": current_mat, "indices": face_indices})

        return {
            "vertices": np.array(vertices, dtype=np.float32),
            "normals": np.array(normals, dtype=np.float32) if normals else None,
            "uvs": np.array(uvs, dtype=np.float32) if uvs else None,
            "faces": faces,
            "materials_content": materials_content,
            "texture_files": texture_files,
            "source_folder": path.parent,
        }

    def _auto_align_mesh_to_world(self, vertices: np.ndarray) -> np.ndarray:

        if len(vertices) < 3:
            return np.eye(3)

        temp_mesh = trimesh.Trimesh(vertices=vertices)
        obb = temp_mesh.bounding_box_oriented
        obb_transform = obb.primitive.transform

        obb_vectors = [obb_transform[:3, 0], obb_transform[:3, 1], obb_transform[:3, 2]]

        world_X = np.array([1, 0, 0])
        world_Y = np.array([0, 1, 0])
        world_Z = np.array([0, 0, 1])

        dots_z = [np.dot(vec, world_Z) for vec in obb_vectors]
        idx_top = np.argmax(np.abs(dots_z))
        vector_top = obb_vectors[idx_top]
        if np.sign(dots_z[idx_top]) < 0:
            vector_top = -vector_top

        remaining_indices = [i for i in range(3) if i != idx_top]
        dots_x = [np.dot(obb_vectors[i], world_X) for i in remaining_indices]
        sub_idx_side = np.argmax(np.abs(dots_x))
        idx_side = remaining_indices[sub_idx_side]
        vector_side = obb_vectors[idx_side]
        if np.sign(dots_x[sub_idx_side]) < 0:
            vector_side = -vector_side

        vector_front = np.cross(vector_side, vector_top)

        current_rotation = np.column_stack((vector_side, vector_top, vector_front))
        return np.linalg.inv(current_rotation)

    def transform_object(
        self,
        obj_data: Dict,
        bounding_box: List[float],
        rotation_quat: np.ndarray,
        center: np.ndarray,
    ):
        vertices = obj_data["vertices"]
        normals = obj_data["normals"]

        vertices_centered = vertices - np.mean(vertices, axis=0)

        auto_align_mat = self._auto_align_mesh_to_world(vertices_centered)
        transformed_v = vertices_centered @ auto_align_mat.T

        min_coords = np.min(transformed_v, axis=0)
        max_coords = np.max(transformed_v, axis=0)
        current_bbox = max_coords - min_coords
        current_bbox = np.where(current_bbox == 0, 1.0, current_bbox)
        scale_factors = np.array(bounding_box) / current_bbox
        transformed_v = transformed_v * scale_factors

        y_rot_mat = quaternion.as_rotation_matrix(self.y_rotation)
        transformed_v = transformed_v @ y_rot_mat.T

        q_input = np.quaternion(*rotation_quat).normalized()
        input_rot_mat = quaternion.as_rotation_matrix(q_input)
        transformed_v = transformed_v @ input_rot_mat.T

        transformed_v = transformed_v + center
        obj_data["vertices"] = transformed_v

        if normals is not None and len(normals) > 0:
            transformed_n = normals @ auto_align_mat.T
            transformed_n = transformed_n @ y_rot_mat.T
            transformed_n = transformed_n @ input_rot_mat.T
            norms = np.linalg.norm(transformed_n, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            obj_data["normals"] = transformed_n / norms

    def _get_world_aabb(self, vertices: np.ndarray) -> Dict[str, float]:
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        return {
            "min_x": min_coords[0],
            "max_x": max_coords[0],
            "min_y": min_coords[1],
            "max_y": max_coords[1],
            "min_z": min_coords[2],
            "max_z": max_coords[2],
        }

    def _enforce_ontop_relationships(
        self, scene_objects: List[Dict], relationships: List[Dict]
    ) -> List[Dict]:
        print("  > Enforcing on_top_of relationships...")
        label_to_obj = {obj["label"]: obj for obj in scene_objects}

        for _ in range(3):
            changes_made = False
            for rel in relationships:
                if rel.get("relation") != "on_top_of":
                    continue

                source_label = rel.get("source", "").capitalize()
                target_label = rel.get("target", "").capitalize()

                top_obj = label_to_obj.get(source_label)
                base_obj = label_to_obj.get(target_label)
                if not top_obj or not base_obj:
                    continue

                top_aabb = self._get_world_aabb(top_obj["vertices"])
                base_aabb = self._get_world_aabb(base_obj["vertices"])

                base_center_x = (base_aabb["min_x"] + base_aabb["max_x"]) / 2.0
                base_center_z = (base_aabb["min_z"] + base_aabb["max_z"]) / 2.0
                top_center_x = (top_aabb["min_x"] + top_aabb["max_x"]) / 2.0
                top_center_z = (top_aabb["min_z"] + top_aabb["max_z"]) / 2.0

                offset_x = base_center_x - top_center_x
                offset_z = base_center_z - top_center_z

                target_y = base_aabb["max_y"]
                offset_y = target_y - top_aabb["min_y"]

                if abs(offset_x) > 1e-4 or abs(offset_y) > 1e-4 or abs(offset_z) > 1e-4:
                    offset_vec = np.array([offset_x, offset_y, offset_z])
                    top_obj["vertices"] += offset_vec
                    top_obj["center"] = (
                        np.array(top_obj["center"]) + offset_vec
                    ).tolist()
                    changes_made = True
            if not changes_made:
                break
        return scene_objects

    def _align_all_to_common_floor(
        self, scene_objects: List[Dict]
    ) -> Tuple[List[Dict], float]:
        print("  > Applying gravity (floor alignment)...")
        if not scene_objects:
            return scene_objects, 0.0

        all_vertices_list = [obj["vertices"] for obj in scene_objects]
        all_v = np.vstack(all_vertices_list)
        global_floor_y = np.min(all_v[:, 1])

        for obj in scene_objects:
            if "Lighting" in obj["label"]:
                continue

            current_min_y = np.min(obj["vertices"][:, 1])
            shift_y = global_floor_y - current_min_y

            if abs(shift_y) > 1e-5:
                shift_vec = np.array([0.0, shift_y, 0.0])
                obj["vertices"] += shift_vec
                obj["center"] = (np.array(obj["center"]) + shift_vec).tolist()

        return scene_objects, global_floor_y

    def _realign_objects_to_key(self, objects_data: List[Dict]) -> List[Dict]:
        if not objects_data:
            return objects_data

        volumes = [
            obj["bounding_box"][0] * obj["bounding_box"][1] * obj["bounding_box"][2]
            for obj in objects_data
        ]
        key_idx = np.argmax(volumes)
        key_obj = objects_data[key_idx]

        key_quat = np.quaternion(*key_obj["rotation"]).normalized()
        correction_quat = key_quat.inverse()

        aligned_objects = []
        for obj in objects_data:
            curr_quat = np.quaternion(*obj["rotation"]).normalized()
            new_quat = correction_quat * curr_quat
            curr_center = np.array(obj["center"])
            new_center = quaternion.rotate_vectors(correction_quat, curr_center)

            new_obj = obj.copy()
            new_obj["rotation"] = [new_quat.w, new_quat.x, new_quat.y, new_quat.z]
            new_obj["center"] = new_center.tolist()
            aligned_objects.append(new_obj)
        return aligned_objects

    def load_scene_from_json(
        self,
        json_path: str,
        obj_folder: str,
        relationships: Optional[List[Dict]] = None,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj_folder_path = Path(obj_folder)

        for scene_id, raw_objects in data.items():
            print(f"\nLoading scene: {scene_id}")
            aligned_metadata = self._realign_objects_to_key(raw_objects)
            scene_objects = []

            scene_mtl_lines = []
            scene_texture_files = []

            for obj_data in aligned_metadata:
                label = obj_data["label"]
                obj_file = obj_folder_path / f"{label}.obj"
                if not obj_file.exists():
                    continue

                try:
                    obj_full = self.load_obj_with_mtl(str(obj_file), label)

                    self.transform_object(
                        obj_full,
                        obj_data["bounding_box"],
                        np.array(obj_data["rotation"]),
                        np.array(obj_data["center"]),
                    )

                    obj_full["label"] = label
                    obj_full["rotation"] = obj_data["rotation"]
                    obj_full["center"] = obj_data["center"]
                    obj_full["bounding_box"] = obj_data["bounding_box"]

                    scene_objects.append(obj_full)
                    scene_mtl_lines.extend(obj_full["materials_content"])

                    for tex_file in obj_full.get("texture_files", []):
                        source_path = obj_full["source_folder"] / tex_file
                        if source_path.exists():
                            scene_texture_files.append((source_path, tex_file))
                        else:
                            print(f"  [Warning] Texture not found: {source_path}")

                except Exception as e:
                    print(f"  Error loading {label}: {e}")

            scene_objects, floor_y = self._align_all_to_common_floor(scene_objects)
            if relationships:
                scene_objects = self._enforce_ontop_relationships(
                    scene_objects, relationships
                )

            self.scenes[scene_id] = scene_objects
            self.scene_floor_heights[scene_id] = floor_y
            self.scene_materials[scene_id] = scene_mtl_lines
            self.scene_textures[scene_id] = scene_texture_files

            print(
                f"Scene {scene_id}: Loaded {len(scene_objects)} objects, {len(scene_texture_files)} textures."
            )

    def export_scene(self, scene_id: str, output_path: str, include_floor: bool = True):
        if scene_id not in self.scenes:
            return
        objects = self.scenes[scene_id]
        floor_height = self.scene_floor_heights.get(scene_id, 0.0)
        mtl_lines = self.scene_materials.get(scene_id, [])
        texture_files = self.scene_textures.get(scene_id, [])

        out_obj_path = Path(output_path)
        out_mtl_path = out_obj_path.with_suffix(".mtl")
        out_dir = out_obj_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        mtl_filename = out_mtl_path.name

        for src_path, tex_filename in texture_files:
            dst_path = out_dir / tex_filename
            try:
                shutil.copy2(src_path, dst_path)
                print(f"  Copied texture: {tex_filename}")
            except Exception as e:
                print(f"  Warning: Could not copy texture {tex_filename}: {e}")

        with open(out_mtl_path, "w", encoding="utf-8") as f_mtl:
            f_mtl.write(f"# Material Lib for Scene {scene_id}\n")
            for line in mtl_lines:
                f_mtl.write(line + "\n")

            if include_floor:
                f_mtl.write("\nnewmtl Floor_Mat\n")
                f_mtl.write("Kd 0.8 0.8 0.8\n")
                f_mtl.write("Ns 10.0\n")

        with open(out_obj_path, "w", encoding="utf-8") as f:
            f.write(f"# 3D Scene: {scene_id}\n")
            f.write(f"mtllib {mtl_filename}\n")

            v_offset = 0
            vt_offset = 0
            vn_offset = 0

            if include_floor:
                f.write(f"o Floor\n")
                w, d = self.floor_size[0] / 2, self.floor_size[1] / 2
                y = floor_height
                f.write(f"v {-w:.4f} {y:.4f} {-d:.4f}\n")
                f.write(f"v {w:.4f} {y:.4f} {-d:.4f}\n")
                f.write(f"v {w:.4f} {y:.4f} {d:.4f}\n")
                f.write(f"v {-w:.4f} {y:.4f} {d:.4f}\n")
                f.write("vn 0.0 1.0 0.0\n")
                f.write("usemtl Floor_Mat\n")
                f.write(
                    f"f {1+v_offset}//{1+vn_offset} {2+v_offset}//{1+vn_offset} {3+v_offset}//{1+vn_offset} {4+v_offset}//{1+vn_offset}\n\n"
                )

                v_offset += 4
                vn_offset += 1

            for obj in objects:
                f.write(f"o {obj['label']}\n")

                for v in obj["vertices"]:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                if obj["uvs"] is not None:
                    for vt in obj["uvs"]:
                        f.write(f"vt {vt[0]:.6f} {vt[1]:.6f}\n")

                if obj["normals"] is not None:
                    for vn in obj["normals"]:
                        f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

                current_w_mat = None
                for face_data in obj["faces"]:
                    mat_name = face_data["material"]
                    if mat_name and mat_name != current_w_mat:
                        f.write(f"usemtl {mat_name}\n")
                        current_w_mat = mat_name

                    face_str_parts = []
                    for idx_tuple in face_data["indices"]:
                        v_idx, vt_idx, vn_idx = idx_tuple

                        s_v = str(v_idx + v_offset + 1)
                        s_vt = str(vt_idx + vt_offset + 1) if vt_idx is not None else ""
                        s_vn = str(vn_idx + vn_offset + 1) if vn_idx is not None else ""

                        if not s_vt and not s_vn:
                            face_str_parts.append(s_v)
                        elif not s_vt and s_vn:
                            face_str_parts.append(f"{s_v}//{s_vn}")
                        else:
                            face_str_parts.append(f"{s_v}/{s_vt}/{s_vn}")

                    f.write(f"f {' '.join(face_str_parts)}\n")

                v_offset += len(obj["vertices"])
                if obj["uvs"] is not None:
                    vt_offset += len(obj["uvs"])
                if obj["normals"] is not None:
                    vn_offset += len(obj["normals"])

                f.write("\n")

        print(f"✓ Exported: {output_path} (with MTL and Textures)")

    def export_all_scenes(self, output_folder: str, include_floor: bool = True):
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        for scene_id in self.scenes.keys():
            safe_name = scene_id.replace("/", "_").replace("\\", "_")
            self.export_scene(
                scene_id, str(output_path / f"{safe_name}.obj"), include_floor
            )
