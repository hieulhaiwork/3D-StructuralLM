import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the VQ-VAE components from train_chamfer.py
from train_chamfer import (
    PointCloudVQVAE, 
    FilteredDataset,
    ResidualLayer,
    ResidualStack,
    PointCloudEncoder,
    VectorQuantizer
)

def load_model(checkpoint_path, device):
    """Load the trained VQ-VAE model from checkpoint"""
    
    # Model hyperparameters (should match training)
    h_dim = 256
    res_h_dim = 128
    n_res_layers = 3
    n_embeddings = 2048
    embedding_dim = 64
    beta = 0.25
    
    # Create model
    model = PointCloudVQVAE(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"Training phase: {checkpoint.get('phase', 'unknown')}")
            print(f"Final loss: {checkpoint.get('loss', 'unknown'):.4f}")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print("Model loaded (direct state dict)")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    model.to(device)
    model.eval()
    return model

def encode_point_cloud(model, point_cloud, device):
    """
    Encode a single point cloud using the VQ-VAE encoder
    
    Args:
        model: Trained VQ-VAE model
        point_cloud: numpy array of shape (num_points, 3)
        device: torch device
    
    Returns:
        encoded_features: Encoded features before quantization
        quantized_features: Quantized features
        encoding_indices: Indices of selected codebook vectors
        perplexity: Perplexity value
    """
    
    # Convert to tensor and add batch dimension
    if isinstance(point_cloud, np.ndarray):
        point_cloud = torch.from_numpy(point_cloud).float()
    
    if len(point_cloud.shape) == 2:
        point_cloud = point_cloud.unsqueeze(0)  # Add batch dimension
    
    point_cloud = point_cloud.to(device)
    
    with torch.no_grad():
        # Convert to (batch, 3, num_points)
        x = point_cloud.permute(0, 2, 1)
        
        # Encode
        z_e = model.encoder(x)  # Encoded features
        z_e = model.pre_quantization_conv(z_e)  # Project to embedding space
        
        # Quantize
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices = model.vector_quantization(z_e)
        
        return z_e, z_q, min_encoding_indices, perplexity

def analyze_encoding(z_e, z_q, indices, perplexity):
    """Analyze and print encoding results"""
    
    print("="*60)
    print("ENCODING ANALYSIS")
    print("="*60)
    
    # Basic shape information
    print(f"Encoded features (z_e) shape: {z_e.shape}")
    print(f"Quantized features (z_q) shape: {z_q.shape}")
    print(f"Encoding indices shape: {indices.shape}")
    
    # Vector length analysis
    batch_size, embedding_dim, num_tokens = z_e.shape
    total_elements = batch_size * embedding_dim * num_tokens
    
    print(f"\nVECTOR DIMENSIONS:")
    print(f"- Batch size: {batch_size}")
    print(f"- Embedding dimension: {embedding_dim}")
    print(f"- Number of tokens: {num_tokens}")
    print(f"- Total elements in encoded vector: {total_elements}")
    print(f"- Total parameters per sample: {embedding_dim * num_tokens}")
    
    # Statistical analysis
    z_e_np = z_e.cpu().numpy()
    z_q_np = z_q.cpu().numpy()
    
    print(f"\nSTATISTICAL ANALYSIS:")
    print(f"Encoded features (z_e):")
    print(f"  - Mean: {np.mean(z_e_np):.6f}")
    print(f"  - Std: {np.std(z_e_np):.6f}")
    print(f"  - Min: {np.min(z_e_np):.6f}")
    print(f"  - Max: {np.max(z_e_np):.6f}")
    
    print(f"Quantized features (z_q):")
    print(f"  - Mean: {np.mean(z_q_np):.6f}")
    print(f"  - Std: {np.std(z_q_np):.6f}")
    print(f"  - Min: {np.min(z_q_np):.6f}")
    print(f"  - Max: {np.max(z_q_np):.6f}")
    
    # Quantization analysis
    indices_np = indices.cpu().numpy().flatten()
    unique_indices = np.unique(indices_np)
    
    print(f"\nQUANTIZATION ANALYSIS:")
    print(f"- Perplexity: {perplexity:.2f}")
    print(f"- Unique codebook indices used: {len(unique_indices)} out of total tokens: {len(indices_np)}")
    print(f"- Codebook utilization: {len(unique_indices)/len(indices_np)*100:.1f}%")
    print(f"- Most frequent indices: {np.bincount(indices_np).argsort()[-5:][::-1]}")
    
    return {
        'shape': z_e.shape,
        'total_elements': total_elements,
        'embedding_dim': embedding_dim,
        'num_tokens': num_tokens,
        'perplexity': perplexity,
        'unique_codes_used': len(unique_indices)
    }

def visualize_encoding(point_cloud, z_e, save_path='logs/encoding_analysis.png'):
    """Visualize original point cloud and encoding heatmap"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original point cloud
    if len(point_cloud.shape) == 3:
        pc = point_cloud[0].cpu().numpy()  # Take first batch
    else:
        pc = point_cloud.cpu().numpy()
    
    ax1.scatter(pc[:, 0], pc[:, 1], c=pc[:, 2], s=1, alpha=0.6)
    ax1.set_title('Original Point Cloud (2D Projection)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    
    # Plot encoding heatmap
    z_e_np = z_e[0].cpu().numpy()  # Take first batch: (embedding_dim, num_tokens)
    im = ax2.imshow(z_e_np, aspect='auto', cmap='viridis')
    ax2.set_title('Encoded Features Heatmap')
    ax2.set_xlabel('Token Index')
    ax2.set_ylabel('Feature Dimension')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Encoding visualization saved to: {save_path}")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_files = [
        'vqvae_pointcloud_final.pth'
    ]
    
    model = None
    for checkpoint_file in checkpoint_files:
        if os.path.exists(checkpoint_file):
            model = load_model(checkpoint_file, device)
            if model is not None:
                print(f"Successfully loaded model from: {checkpoint_file}")
                break
    
    if model is None:
        print("No valid checkpoint found! Please ensure training has completed.")
        return
    
    # Load test data
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    root_dir = "."
    test_dataset = FilteredDataset(root_dir, num_points=1024, split='train', max_samples=5)
    
    if len(test_dataset) == 0:
        print("No test data found!")
        return
    
    # Test encoding on multiple samples
    print(f"\nTesting encoding on {len(test_dataset)} samples...")
    
    all_results = []
    
    for i in range(min(3, len(test_dataset))):  # Test first 3 samples
        print(f"\n{'='*40}")
        print(f"SAMPLE {i+1}: {test_dataset.names[i] if i < len(test_dataset.names) else f'sample_{i}'}")
        print(f"{'='*40}")
        
        # Get sample
        point_cloud, label, name = test_dataset[i]
        print(f"Point cloud shape: {point_cloud.shape}")
        print(f"Label: {label.item()}")
        print(f"Name: {name}")
        
        # Encode
        z_e, z_q, indices, perplexity = encode_point_cloud(model, point_cloud, device)
        
        # Analyze
        result = analyze_encoding(z_e, z_q, indices, perplexity)
        result['sample_name'] = name
        all_results.append(result)
        
        # Visualize first sample
        if i == 0:
            visualize_encoding(point_cloud, z_e, f'logs/encoding_sample_{i}.png')
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_perplexity = np.mean([r['perplexity'].cpu().item() if hasattr(r['perplexity'], 'cpu') else r['perplexity'] for r in all_results])
    avg_unique_codes = np.mean([r['unique_codes_used'] for r in all_results])
    
    print(f"Average perplexity across samples: {avg_perplexity:.2f}")
    print(f"Average unique codes used: {avg_unique_codes:.1f}")
    print(f"Encoding vector dimensions: {all_results[0]['embedding_dim']} x {all_results[0]['num_tokens']}")
    print(f"Total encoding size per sample: {all_results[0]['total_elements']} values")
    
    # Compression ratio
    original_size = 1024 * 3  # 1024 points * 3 coordinates
    compressed_size = all_results[0]['total_elements']
    compression_ratio = original_size / compressed_size
    
    print(f"\nCOMPRESSION ANALYSIS:")
    print(f"Original point cloud size: {original_size} values (1024 points × 3 coords)")
    print(f"Compressed representation size: {compressed_size} values ({all_results[0]['embedding_dim']} × {all_results[0]['num_tokens']})")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"Space savings: {(1 - compressed_size/original_size)*100:.1f}%")

if __name__ == "__main__":
    main()