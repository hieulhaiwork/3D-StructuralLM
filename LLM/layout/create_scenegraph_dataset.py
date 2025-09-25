# file: create_scenegraph_dataset.py

import torch
import numpy as np
import trimesh
from tqdm import tqdm
from typing import Dict, List, Any

import configs as cfg
from utils import (
    classify_cabinet_shelf_desk,
    get_pose,
    generate_relationships,
    generate_caption,
    save_data_to_json,
    quantize_pose 
)

def process_glb_to_scenegraph(scene_path: cfg.Path) -> Dict[str, Any]:
    """
    Hàm chính để xử lý một file GLB và chuyển nó thành cấu trúc Scene Graph hoàn chỉnh.
    """
    try:
        scene = trimesh.load(str(scene_path), force='scene')
    except Exception as e:
        print(f"\nLỗi khi tải scene {scene_path.name}: {e}")
        return None

    raw_entities = []
    
    # Duyệt qua các node hình học trong scene
    for node_name in scene.graph.nodes_geometry:
        _, geometry_name = scene.graph[node_name]
        if geometry_name is None:
            continue
            
        mesh = scene.geometry.get(geometry_name)
        
        # Làm sạch tên để xác định lớp đối tượng
        clean_name = geometry_name.split('.obj')[0] if '.obj' in geometry_name else geometry_name
        initial_class = next((cat for cat in cfg.OBJECT_CLASSES if clean_name.startswith(cat)), None)
        
        if initial_class and isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > cfg.MIN_VERTICES_THRESHOLD:
            pose_data, bounds_data = get_pose(mesh)
            if pose_data is None:
                continue
            
            final_class = initial_class
            if initial_class == "Cabinet":
                scale_vector = np.array(pose_data["scale"])
                final_class = classify_cabinet_shelf_desk(
                    scale_vector, 
                    cfg.TABLE_HEIGHT_THRESHOLD
                )

            quantized_pose = quantize_pose(pose_data, precision=3)

            # Tạo một thực thể (entity)
            entity = {
                "id": clean_name,
                "class": final_class,
                "pose": quantized_pose,
                "_bounds": bounds_data
            }

            if final_class in cfg.OBJECT_CLASSES:
                raw_entities.append(entity)

    if not raw_entities:
        return None
    
    # --- TẠO ID ĐƠN GIẢN ---
    class_counters = {}
    final_entities = []
    for entity in raw_entities:
        class_name_lower = entity["class"].lower()
        
        # Tăng bộ đếm cho lớp này
        current_count = class_counters.get(class_name_lower, 0) + 1
        class_counters[class_name_lower] = current_count
        
        # Tạo ID mới
        new_id = f"{class_name_lower}_{current_count}"

        # Tạo entity cuối cùng với ID mới
        final_entity = {
            "id": new_id,
            "class": entity["class"],
            "description": f"a {entity['class'].lower().replace('_', ' ')}",
            "pose": entity["pose"],
            "_bounds": entity["_bounds"]
        }
        final_entities.append(final_entity)
    
    # Nạp cấu hình suy luận vào một dict để dễ truyền đi
    relationship_config = {
        'FLOOR_Z_THRESHOLD': cfg.FLOOR_Z_THRESHOLD,
        'NEXT_TO_DISTANCE_THRESHOLD': cfg.NEXT_TO_DISTANCE_THRESHOLD,
        'ALIGNMENT_AXIS_THRESHOLD': cfg.ALIGNMENT_AXIS_THRESHOLD
    }
    
    # Suy luận mối quan hệ giữa các thực thể
    relationships = generate_relationships(final_entities, relationship_config)
    
    # Dọn dẹp key tạm thời "_bounds"
    for e in final_entities:
        del e["_bounds"]
        
    # Tạo cấu trúc scene graph cuối cùng
    scene_graph = {
        "scene_info": {
            "id": scene_path.stem.replace('_full', ''),
            "source": "3D-FRONT"
        },
        "entities": final_entities,
        "relationships": relationships
    }
    
    return scene_graph

def main():
    """Hàm điều phối pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng device: {device}")
    
    final_dataset = []

    for scene_filename in tqdm(cfg.SCENE_FILE_PATHS, desc="Tạo Scene Graph"):
        scene_path = cfg.BASE_DATASET_PATH / scene_filename
        
        scene_graph = process_glb_to_scenegraph(scene_path)
        
        if scene_graph and scene_graph.get('entities'):
            caption = generate_caption(scene_graph['entities'])
            llm_sample = {"text": caption, "scene_graph": scene_graph}
            final_dataset.append(llm_sample)
            
    if final_dataset:
        save_data_to_json(final_dataset, cfg.OUTPUT_FILE_PATH)
    else:
        print("Không có dữ liệu hợp lệ nào được xử lý để lưu.")

    print("--- HOÀN TẤT ---")

if __name__ == "__main__":
    main()