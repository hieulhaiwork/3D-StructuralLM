# file: create_latent_dataset.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple

# Import VQ-VAE components from train_chamfer.py
class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim) for _ in range(n_res_layers)])

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)

class PointCloudEncoder(nn.Module):
    def __init__(self, h_dim, n_res_layers, res_h_dim):
        super(PointCloudEncoder, self).__init__()
        self.h_dim = h_dim 

        self.conv_stack = nn.Sequential(
            nn.Conv1d(3, h_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim // 4, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1), 
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        z_q = z + (z_q - z).detach()
        
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        z_q = z_q.permute(0, 2, 1).contiguous()
        
        return loss, z_q, perplexity, min_encodings, min_encoding_indices.squeeze()

class LatentDatasetConfig:
    """Configuration for latent dataset creation"""
    # Paths
    BASE_DATASET_PATH = Path("./")
    OUTPUT_FILE_PATH = Path("./latent_dataset_for_llm_training.json")
    VQVAE_MODEL_PATH = Path("./vqvae_pointcloud_final.pth")
    
    # Object classes to extract
    OBJECT_CLASSES = ["Bed", "Cabinet", "Table"]
    
    # Processing parameters
    MIN_VERTICES_THRESHOLD = 100
    NUM_POINTS = 1024  # Number of points to sample from each object
    TABLE_HEIGHT_THRESHOLD = 0.5  # meters
    
    # VQ-VAE model parameters (should match training config)
    H_DIM = 256
    RES_H_DIM = 128
    N_RES_LAYERS = 3
    N_EMBEDDINGS = 2048
    EMBEDDING_DIM = 64
    BETA = 0.25

def get_bedroom_scene_paths(base_path: Path) -> List[str]:
    """Get all bedroom .glb file paths"""
    try:
        all_full_glb_files = base_path.rglob("*_full.glb")
        scene_file_paths = [
            p.as_posix() 
            for p in all_full_glb_files 
            if "bedroom" in p.name.lower()
        ]
        return scene_file_paths
    except FileNotFoundError:
        print(f"ERROR: Dataset directory not found at '{base_path}'. Please check the path.")
        return []

def classify_cabinet_shelf_desk(scale: np.ndarray, threshold: float) -> str:
    """Classify object as Cabinet or Table based on height"""
    height = scale[2]
    return "Table" if height < threshold else "Cabinet"

def sample_points_from_mesh(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Sample points from mesh surface"""
    if len(mesh.vertices) < num_points:
        # If mesh has fewer vertices than needed, sample with replacement
        indices = np.random.choice(len(mesh.vertices), num_points, replace=True)
        points = mesh.vertices[indices]
    else:
        # Sample points from mesh surface
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    return points.astype(np.float32)

def extract_objects_from_scene(scene_path: Path, config: LatentDatasetConfig) -> Tuple[List[Dict], str]:
    """
    Extract objects from a GLB scene file.
    Returns: (list of objects with their point clouds, scene caption)
    """
    try:
        scene = trimesh.load(str(scene_path), force='scene')
    except Exception as e:
        print(f"Error loading scene {scene_path.name}: {e}")
        return [], ""

    objects = []
    object_counts = {}
    
    # Process each geometry node in the scene
    for node_name in scene.graph.nodes_geometry:
        _, geometry_name = scene.graph[node_name]
        if geometry_name is None:
            continue
            
        mesh = scene.geometry.get(geometry_name)
        
        # Clean name to identify object class
        clean_name = geometry_name.split('.obj')[0] if '.obj' in geometry_name else geometry_name
        initial_class = next((cat for cat in config.OBJECT_CLASSES if clean_name.startswith(cat)), None)
        
        if initial_class and isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) > config.MIN_VERTICES_THRESHOLD:
            # Get mesh bounds for classification
            bounds = mesh.bounds
            scale = bounds[1] - bounds[0]
            
            final_class = initial_class
            if initial_class == "Cabinet":
                final_class = classify_cabinet_shelf_desk(scale, config.TABLE_HEIGHT_THRESHOLD)

            # Sample points from mesh
            points = sample_points_from_mesh(mesh, config.NUM_POINTS)
            
            # Count objects for caption generation
            class_name_lower = final_class.lower()
            object_counts[class_name_lower] = object_counts.get(class_name_lower, 0) + 1
            
            objects.append({
                "class": final_class,
                "points": points  # Shape: (num_points, 3)
            })
    
    # Generate caption
    if not object_counts:
        caption = "An empty room."
    else:
        parts = [f"{count} {name}s" if count > 1 else f"a {name}" 
                for name, count in object_counts.items()]
        
        if len(parts) == 1:
            caption = f"A room containing {parts[0]}."
        else:
            caption = f"A room containing {', '.join(parts[:-1])} and {parts[-1]}."
    
    return objects, caption

def load_vqvae_encoder(model_path: Path, config: LatentDatasetConfig, device: torch.device):
    """Load the trained VQ-VAE encoder"""
    # Create encoder
    encoder = PointCloudEncoder(config.H_DIM, config.N_RES_LAYERS, config.RES_H_DIM)
    pre_quantization_conv = nn.Conv1d(config.H_DIM, config.EMBEDDING_DIM, kernel_size=1, stride=1)
    vector_quantizer = VectorQuantizer(config.N_EMBEDDINGS, config.EMBEDDING_DIM, config.BETA)
    
    # Load full model state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Extract encoder weights
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    pre_quant_state = {k.replace('pre_quantization_conv.', ''): v for k, v in state_dict.items() if k.startswith('pre_quantization_conv.')}
    vq_state = {k.replace('vector_quantization.', ''): v for k, v in state_dict.items() if k.startswith('vector_quantization.')}
    
    encoder.load_state_dict(encoder_state)
    pre_quantization_conv.load_state_dict(pre_quant_state)
    vector_quantizer.load_state_dict(vq_state)
    
    # Move to device and set to eval mode
    encoder = encoder.to(device).eval()
    pre_quantization_conv = pre_quantization_conv.to(device).eval()
    vector_quantizer = vector_quantizer.to(device).eval()
    
    return encoder, pre_quantization_conv, vector_quantizer

def encode_object_to_indices(points: np.ndarray, encoder, pre_quant_conv, vector_quantizer, device: torch.device) -> List[int]:
    """
    Encode a single object's point cloud to quantized indices.
    
    Args:
        points: numpy array of shape (num_points, 3)
        encoder, pre_quant_conv, vector_quantizer: model components
        device: torch device
    
    Returns:
        List of 64 integers (indices in codebook)
    """
    with torch.no_grad():
        # Convert to tensor and add batch dimension
        points_tensor = torch.from_numpy(points).unsqueeze(0).to(device)  # (1, num_points, 3)
        
        # Permute to (batch, channels, num_points) format for conv1d
        points_tensor = points_tensor.permute(0, 2, 1)  # (1, 3, num_points)
        
        # Encode
        z_e = encoder(points_tensor)  # (1, h_dim, 64)
        z_e = pre_quant_conv(z_e)  # (1, embedding_dim, 64)
        
        # Quantize
        _, z_q, _, _, indices = vector_quantizer(z_e)  # indices shape: (64,)
        
        # Convert to list of integers
        indices_list = indices.cpu().numpy().tolist()
        
        return indices_list

def create_latent_dataset():
    """Main function to create the latent dataset"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = LatentDatasetConfig()
    
    # Get scene file paths
    scene_paths = get_bedroom_scene_paths(config.BASE_DATASET_PATH)
    if not scene_paths:
        print("No bedroom scenes found!")
        return
    
    print(f"Found {len(scene_paths)} bedroom scenes")
    
    # Load VQ-VAE encoder
    print("Loading VQ-VAE encoder...")
    try:
        encoder, pre_quant_conv, vector_quantizer = load_vqvae_encoder(
            config.VQVAE_MODEL_PATH, config, device
        )
        print("VQ-VAE encoder loaded successfully!")
    except Exception as e:
        print(f"Error loading VQ-VAE model: {e}")
        return
    
    # Process each scene
    dataset = []
    
    for scene_path_str in tqdm(scene_paths, desc="Processing scenes"):
        scene_path = Path(scene_path_str)
        
        # Extract objects from scene
        objects, caption = extract_objects_from_scene(scene_path, config)
        
        if not objects:
            continue
        
        # Encode each object to indices
        predicted_indices = {}
        
        for obj in objects:
            class_name = obj["class"].lower()
            points = obj["points"]
            
            # Encode object to indices
            indices = encode_object_to_indices(
                points, encoder, pre_quant_conv, vector_quantizer, device
            )
            
            # Handle multiple objects of same class by numbering them
            base_key = class_name
            counter = 1
            key = base_key
            while key in predicted_indices:
                counter += 1
                key = f"{base_key}_{counter}"
            
            predicted_indices[key] = indices
        
        # Create dataset sample
        sample = {
            "text": caption,
            "predicted_indices": predicted_indices
        }
        
        dataset.append(sample)
    
    # Save dataset
    if dataset:
        with open(config.OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully created latent dataset with {len(dataset)} samples!")
        print(f"Saved to: {config.OUTPUT_FILE_PATH}")
        
        # Print some statistics
        total_objects = sum(len(sample["predicted_indices"]) for sample in dataset)
        print(f"Total objects encoded: {total_objects}")
        
        # Show a sample
        if dataset:
            print("\nSample from dataset:")
            sample = dataset[0]
            print(f"Text: {sample['text']}")
            print(f"Objects: {list(sample['predicted_indices'].keys())}")
            print(f"Indices for first object: {list(sample['predicted_indices'].values())[0][:10]}... (showing first 10)")
    else:
        print("No valid data processed!")

if __name__ == "__main__":
    create_latent_dataset()