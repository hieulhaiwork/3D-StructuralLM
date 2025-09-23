# file: main.py

import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
import trimesh
import torch

from vqvae_encoder import VQVAEEncoderLoader 
from utils import (
    load_glb_as_point_cloud,
    normalize_point_cloud,
    generate_caption_from_scene_data,
    save_training_data
)

# --- Cấu hình ---
BASE_DATA_PATH = Path("./")
SCENE_FILES_TO_PROCESS = [
    "dataset/3D-FRONT-TEST-SCENE/deaa4ac0-8d13-423f-930f-78fb25825e8d/Bedroom-5088_full.glb",
    "dataset/3D-FRONT-TEST-SCENE/decb93da-ff77-49c3-9cad-fbb3596d36e8/SecondBedroom-11739_full.glb",
    "dataset/3D-FRONT-TEST-SCENE/dee82180-497b-489f-8554-cb27be17300f/Bedroom-46392_full.glb",
    "dataset/3D-FRONT-TEST-SCENE/deeab9a9-dc28-4fff-a2b0-6e4eec1b60dc/MasterBedroom-11650_full.glb",
    "dataset/3D-FRONT-TEST-SCENE/def02b2c-2ec1-40cf-95d3-f3e095c8043e/SecondBedroom-4035_full.glb",
    "dataset/3D-FRONT-TEST-SCENE/df0e6d6c-5251-43ab-8334-e67e11ffc697/SecondBedroom-155750_full.glb",
    "dataset/3D-FRONT-TEST-SCENE/df1d3950-d775-41ae-ab88-261156e7b5b9/MasterBedroom-1095_full.glb",
    "dataset/3D-FRONT-TEST-SCENE/df1d3950-d775-41ae-ab88-261156e7b5b9/SecondBedroom-2941_full.glb"
]
TARGET_OBJECT_CLASSES = ["Bed", "Table", "Cabinet_Shelf_Desk"]
NUM_POINTS_PER_OBJECT = 2048
OUTPUT_FILE = Path("./training_data_final.json")


def get_pose_from_baked_mesh(mesh: trimesh.Trimesh) -> Dict[str, List[float]]:
    """Suy ngược pose từ một mesh có vertex đã ở tọa độ thế giới."""
    if not mesh.vertices.size:
        return None
        
    position = mesh.bounds.mean(axis=0)
    scale = mesh.bounds[1] - mesh.bounds[0]
    rotation_quaternion = [1.0, 0.0, 0.0, 0.0]
    
    return {
        "position": position.tolist(),
        "rotation_quaternion": rotation_quaternion,
        "scale": scale.tolist()
    }

def process_full_scene_glb_baked(scene_path: Path, encoder: Any) -> Dict[str, Any]:
    """Xử lý file GLB có vertex đã được "nướng" vào tọa độ thế giới."""
    if not scene_path.exists(): return None
    try:
        scene = trimesh.load(str(scene_path), force='scene')
    except Exception as e:
        print(f"Error loading scene {scene_path.name}: {e}")
        return None

    processed_objects = []
    
    # Duyệt qua các node có geometry
    for node_name in scene.graph.nodes_geometry:
        # Lấy geometry_name từ node. Dù transform là identity, chúng ta vẫn cần nó
        # để lấy geometry_name một cách chính xác.
        _, geometry_name = scene.graph[node_name]
        
        if geometry_name is None:
            continue
            
        mesh = scene.geometry.get(geometry_name)
        
        # Làm sạch tên để so sánh
        clean_name = geometry_name.split('.obj')[0] if '.obj' in geometry_name else geometry_name
        
        obj_class = next((cat for cat in TARGET_OBJECT_CLASSES if clean_name.startswith(cat)), None)
        
        # Kiểm tra điều kiện một cách cẩn thận
        if obj_class and isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > 0:
            pose_data = get_pose_from_baked_mesh(mesh)
            if pose_data is None:
                continue

            point_cloud = mesh.sample(NUM_POINTS_PER_OBJECT)
            normalized_pc = normalize_point_cloud(point_cloud)
            latent_code_array = encoder.encode(normalized_pc)
            latent_code_list = latent_code_array.tolist()
            
            processed_objects.append({
                "class": obj_class,
                "jid": clean_name, # Dùng tên đã được làm sạch
                "pose": pose_data,
                "latent_code": latent_code_list 
            })

    if not processed_objects: return None
    return {"scene_id": scene_path.stem.replace('_full', ''), "objects": processed_objects}

def main():
    """Hàm chính cuối cùng."""
    # 1. Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Đường dẫn đến model đã huấn luyện
    model_path = Path("./vqvae_pointcloud_epoch_240.pth")

    # 3. Khởi tạo Encoder thật thay vì DummyEncoder
    try:
        encoder = VQVAEEncoderLoader(model_path, device)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error initializing VQVAEEncoderLoader: {e}")
        print("Aborting pipeline.")
        return
    
    final_training_data = []

    for scene_filename in tqdm(SCENE_FILES_TO_PROCESS, desc="Processing Scenes"):
        scene_path = BASE_DATA_PATH / scene_filename
        
        scene_data = process_full_scene_glb_baked(scene_path, encoder)
        
        if scene_data and scene_data['objects']:
            caption = generate_caption_from_scene_data(scene_data)
            llm_sample = {"text": caption, "scene_description": scene_data}
            final_training_data.append(llm_sample)
            
    if final_training_data:
        save_training_data(final_training_data, OUTPUT_FILE)
    else:
        print("No valid scenes or objects were processed.")

    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()