# file: sg_utils.py

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import trimesh
from itertools import combinations

class NumpyArrayEncoder(json.JSONEncoder):
    """
    Một bộ mã hóa JSON tùy chỉnh để xử lý các đối tượng numpy.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Chuyển ndarray thành list
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj) # Chuyển các kiểu int của numpy thành int của Python
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj) # Chuyển các kiểu float của numpy thành float của Python
        if isinstance(obj, (np.bool_)):
            return bool(obj) # Chuyển kiểu bool của numpy thành bool của Python
        return super(NumpyArrayEncoder, self).default(obj)
    
def classify_cabinet_shelf_desk(scale: np.ndarray, threshold: float) -> str:
    """
    Phân loại một vật thể là "Cabinet" hoặc "Table" dựa trên chiều cao của nó.
    """
    height = scale[2]

    if height < threshold:
        return "Table"
    else:
        return "Cabinet"

def get_pose(mesh: trimesh.Trimesh) -> Optional[Tuple[Dict[str, List[float]], np.ndarray]]:
    """
    Suy ngược pose từ một mesh có vertex đã ở tọa độ thế giới.
    Trả về một tuple chứa (pose_dictionary, raw_bounds).
    """
    if not mesh.vertices.size:
        return None
        
    bounds = mesh.bounds
    position = bounds.mean(axis=0)
    scale = bounds[1] - bounds[0]

    # Vì tập 3D-FRONT đã được chuẩn hóa về mặt hướng, ta có thể sử dụng đơn vị quaternion
    rotation_quaternion = [1.0, 0.0, 0.0, 0.0]

    pose = {
        "position": position.tolist(),
        "rotation_quaternion": rotation_quaternion,
        "scale": scale.tolist()
    }
    return pose, bounds

def generate_relationships(entities: List[Dict], config: Dict) -> List[Dict]:
    """
    Từ danh sách các thực thể, suy luận ra các mối quan hệ không gian dựa trên cấu hình.
    """
    relationships = []
    
    # 1. Suy luận quan hệ "on_surface" (trên sàn)
    on_floor_ids = [
        e["id"] for e in entities 
        if abs(e["_bounds"][0][2]) < config['FLOOR_Z_THRESHOLD']
    ]
    if on_floor_ids:
        relationships.append({"type": "on_surface", "subjects": on_floor_ids, "target": "floor"})

    # Duyệt qua tất cả các cặp đối tượng duy nhất
    for entity_a, entity_b in combinations(entities, 2):
        pos_a = np.array(entity_a["pose"]["position"])
        pos_b = np.array(entity_b["pose"]["position"])
        bounds_a = entity_a["_bounds"]
        bounds_b = entity_b["_bounds"]
        
        # 2. Suy luận quan hệ "next_to" (cạnh nhau)
        distance = np.linalg.norm(pos_a - pos_b)
        if distance < config['NEXT_TO_DISTANCE_THRESHOLD']:
            relationships.append({
                "type": "next_to",
                "subjects": [entity_a["id"], entity_b["id"]],
                "params": {"distance": round(float(distance), 3)}
            })

        # 3. Suy luận quan hệ "align_with" (thẳng hàng)
        # Kiểm tra thẳng hàng theo trục X
        if abs(bounds_a.mean(axis=0)[0] - bounds_b.mean(axis=0)[0]) < config['ALIGNMENT_AXIS_THRESHOLD']:
             relationships.append({
                "type": "align_with",
                "subjects": [entity_a["id"], entity_b["id"]],
                "params": {"axis": "x"}
            })
        # Kiểm tra thẳng hàng theo trục Y
        if abs(bounds_a.mean(axis=0)[1] - bounds_b.mean(axis=0)[1]) < config['ALIGNMENT_AXIS_THRESHOLD']:
             relationships.append({
                "type": "align_with",
                "subjects": [entity_a["id"], entity_b["id"]],
                "params": {"axis": "y"}
            })
            
    return relationships

def generate_caption(entities: List[Dict]) -> str:
    """Tạo caption đơn giản từ danh sách thực thể."""
    counts = {}
    for e in entities:
        class_name = e['class'].replace('_', ' ').lower()
        counts[class_name] = counts.get(class_name, 0) + 1

    if not counts: return "An empty room."

    parts = [f"{count} {name}s" if count > 1 else f"a {name}" for name, count in counts.items()]
    
    if len(parts) == 1:
        return f"A room containing {parts[0]}."
    else:
        return f"A room containing {', '.join(parts[:-1])} and {parts[-1]}."

def save_data_to_json(data: List[Dict], output_path: Path):
    """Lưu dữ liệu vào file JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyArrayEncoder)
    print(f"Thành công! Đã lưu {len(data)} mẫu dữ liệu vào {output_path}")

def quantize_pose(pose_data: Dict, precision: int = 3) -> Dict:
    """
    Làm tròn các giá trị position và scale trong pose đến một số chữ số thập phân nhất định.
    """
    pose_data["position"] = [round(p, precision) for p in pose_data["position"]]
    pose_data["scale"] = [round(s, precision) for s in pose_data["scale"]]
    return pose_data