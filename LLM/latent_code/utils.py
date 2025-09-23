import trimesh
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

def load_glb_as_point_cloud(
    glb_path: Path, 
    num_points: int = 2048
) -> Optional[np.ndarray]:
    """Tải file GLB, chuyển thành mesh và lấy mẫu point cloud."""
    if not glb_path.exists():
        print(f"Warning: GLB file not found at {glb_path}")
        return None
    try:
        mesh = trimesh.load(str(glb_path), force='mesh')
        point_cloud = mesh.sample(num_points)
        return point_cloud
    except Exception as e:
        print(f"Error loading or sampling {glb_path}: {e}")
        return None

def normalize_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """Chuẩn hóa point cloud về gốc tọa độ và nằm trong hộp đơn vị."""
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid
    max_dist = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
    if max_dist > 0:
        point_cloud = point_cloud / max_dist
    return point_cloud

def generate_caption_from_scene_data(scene_data: Dict[str, Any]) -> str:
    """Tạo caption đơn giản từ dữ liệu scene đã được trích xuất."""
    object_counts = {}
    for obj in scene_data.get('objects', []):
        class_name = obj['class'].replace('_', ' ').lower()
        object_counts[class_name] = object_counts.get(class_name, 0) + 1

    if not object_counts:
        return "An empty room."

    descriptions = []
    for name, count in object_counts.items():
        if count > 1:
            descriptions.append(f"{count} {name}s")
        else:
            descriptions.append(f"a {name}")
    
    return f"A room containing {', '.join(descriptions[:-1])} and {descriptions[-1]}."

def save_training_data(data: List[Dict[str, Any]], output_path: Path):
    """Lưu dữ liệu huấn luyện đã xử lý vào file JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Successfully saved {len(data)} samples to {output_path}")