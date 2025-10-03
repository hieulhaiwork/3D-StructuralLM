import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pytorch3d.loss import chamfer_distance
from emd import earth_mover_distance

# ==================== VQ-VAE Components ====================

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

        # Tăng năng lực của mạng
        self.h_dim = h_dim 

        self.conv_stack = nn.Sequential(
            # Lớp 1: 1024 -> 512
            nn.Conv1d(3, h_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Lớp 2: 512 -> 256
            nn.Conv1d(h_dim // 4, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Lớp 3: 256 -> 128
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Lớp 4: 128 -> 64
            nn.Conv1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1), 
            # Các lớp Residual để xử lý sâu hơn ở độ phân giải thấp
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        # x shape: (batch, 3, num_points)
        return self.conv_stack(x)

class PointCloudDecoder(nn.Module):
    def __init__(self, embedding_dim, h_dim, n_res_layers, res_h_dim):
        super(PointCloudDecoder, self).__init__()
        self.h_dim = h_dim
        
        # First convert embedding_dim to h_dim
        self.pre_decoder_conv = nn.Conv1d(embedding_dim, h_dim, kernel_size=1, stride=1)
        
        # Input shape: (B, embedding_dim, 64) -> (B, h_dim, 64)
        self.inverse_conv_stack = nn.Sequential(
            # Xử lý sâu ở độ phân giải thấp
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            # Lớp 1: 64 -> 128
            nn.ConvTranspose1d(h_dim, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Lớp 2: 128 -> 256
            nn.ConvTranspose1d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Lớp 3: 256 -> 512
            nn.ConvTranspose1d(h_dim // 2, h_dim // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Lớp 4: 512 -> 1024
            nn.ConvTranspose1d(h_dim // 4, 3, kernel_size=4, stride=2, padding=1)
        )
        # Output shape: (B, 3, 1024)

    def forward(self, x):
        # Convert embedding_dim to h_dim first
        x = self.pre_decoder_conv(x)
        return self.inverse_conv_stack(x)

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        # z shape: (batch, embedding_dim, num_points)
        z = z.permute(0, 2, 1).contiguous()  # (batch, num_points, embedding_dim)
        z_flattened = z.view(-1, self.e_dim)  # (batch*num_points, embedding_dim)
        
        # Compute distances
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        
        # Get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        
        # Compute loss
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        # Preserve gradients
        z_q = z + (z_q - z).detach()
        
        # Perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        # Reshape back
        z_q = z_q.permute(0, 2, 1).contiguous()  # (batch, embedding_dim, num_points)
        
        return loss, z_q, perplexity, min_encodings, min_encoding_indices

class PointCloudVQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta):
        super(PointCloudVQVAE, self).__init__()
        
        self.encoder = PointCloudEncoder(h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv1d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        self.decoder = PointCloudDecoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

    def forward(self, x):
        # x shape: (batch, num_points, 3) -> (batch, 3, num_points)
        x = x.permute(0, 2, 1)
        
        # Encode
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        
        # Quantize
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        # Decode
        x_hat = self.decoder(z_q)
        
        # Back to original format: (batch, 3, num_points) -> (batch, num_points, 3)
        x_hat = x_hat.permute(0, 2, 1)
        
        return embedding_loss, x_hat, perplexity

# ==================== Dataset Class ====================

# ==================== Dataset Class ====================

class FilteredDataset(data.Dataset):
    def __init__(self, root_dir, num_points=256, split='train', max_samples=None):
        self.num_points = num_points
        self.filtered_dir = os.path.join(root_dir, "filtered_data")
        self.max_samples = max_samples if max_samples is not None else float('inf')  # Set to infinity if None
        
        # Load all data from filtered files
        self.data = []
        self.labels = []
        self.names = []
        
        # Class mapping
        self.class_names = ['bed', 'cabinet', 'table']
        self.class_to_label = {name: idx for idx, name in enumerate(self.class_names)}
        
        total_loaded = 0
        
        print(f"Looking for data in: {self.filtered_dir}")
        print(f"Max samples: {'All available' if max_samples is None else max_samples}")
        
        # Iterate through each class folder
        for class_name in self.class_names:
            if total_loaded >= self.max_samples:
                break
                
            class_dir = os.path.join(self.filtered_dir, class_name)
            
            # Check if class directory exists
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist, skipping {class_name}")
                continue
            
            # Load data for this class
            h5_file = os.path.join(class_dir, f"{class_name}.h5")
            id2name_file = os.path.join(class_dir, f"{class_name}_id2name.json")
            
            if not os.path.exists(h5_file):
                print(f"Warning: H5 file {h5_file} missing for {class_name}, skipping")
                continue
                
            if not os.path.exists(id2name_file):
                print(f"Warning: ID2name file {id2name_file} missing for {class_name}, skipping")
                continue
            
            print(f"Loading data from {h5_file}")
            
            # Load data
            try:
                with h5py.File(h5_file, "r") as f:
                    print(f"Available keys in H5 file: {list(f.keys())}")
                    class_data = f["data"][:]
                    print(f"Loaded {len(class_data)} samples from {class_name}")
                    
                    if "label" in f:
                        class_labels = f["label"][:]
                        print(f"Found labels in H5 file for {class_name}")
                    else:
                        # Create labels based on class index
                        class_labels = np.full(len(class_data), self.class_to_label[class_name])
                        print(f"Created labels for {class_name} with value {self.class_to_label[class_name]}")
            except Exception as e:
                print(f"Error loading H5 file for {class_name}: {e}")
                continue
            
            # Load names
            try:
                with open(id2name_file, "r") as f:
                    class_names_list = json.load(f)
                    print(f"Loaded names for {class_name}, type: {type(class_names_list)}")
            except Exception as e:
                print(f"Error loading names file for {class_name}: {e}")
                continue
            
            # Limit samples for this class
            if max_samples is None:
                samples_to_take = len(class_data)  # Take all samples
            else:
                samples_to_take = min(len(class_data), self.max_samples - total_loaded)
            
            # Add data
            self.data.append(class_data[:samples_to_take])
            
            # Add labels - ensure they are 1D and map to correct class indices
            current_labels = class_labels[:samples_to_take]
            if current_labels.ndim > 1:
                current_labels = current_labels.flatten()
            # Map all labels to the correct class index
            current_labels = np.full(len(current_labels), self.class_to_label[class_name])
            self.labels.append(current_labels)
            
            # Handle names (could be list or dict)
            if isinstance(class_names_list, list):
                self.names.extend(class_names_list[:samples_to_take])
            elif isinstance(class_names_list, dict):
                self.names.extend([class_names_list.get(str(i), f"{class_name}_{i}") for i in range(samples_to_take)])
            else:
                # Fallback: create names
                self.names.extend([f"{class_name}_{i}" for i in range(samples_to_take)])
            
            total_loaded += samples_to_take
            print(f"Added {samples_to_take} samples from {class_name}. Total: {total_loaded}")
        
        # Concatenate all data
        if self.data:
            self.data = np.concatenate(self.data, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            
            # Ensure labels are 1D integers
            self.labels = self.labels.flatten().astype(int)
        else:
            print("No data loaded!")
            self.data = np.array([])
            self.labels = np.array([])
            self.names = []
        
        # No final truncation when max_samples is None
        if max_samples is not None and len(self.data) > max_samples:
            self.data = self.data[:max_samples]
            self.labels = self.labels[:max_samples]
            self.names = self.names[:max_samples]
        
        print(f"\n=== Final Dataset Info ===")
        print(f"Total loaded: {len(self.data)} samples")
        print(f"Data shape: {self.data.shape if len(self.data) > 0 else 'Empty'}")
        print(f"Labels shape: {self.labels.shape if len(self.labels) > 0 else 'Empty'}")
        print(f"Labels type: {type(self.labels)}")
        print(f"Number of names: {len(self.names)}")
        
        if len(self.labels) > 0:
            try:
                unique_labels = np.unique(self.labels)
                print(f"Unique labels: {unique_labels}")
                label_counts = np.bincount(self.labels)
                print(f"Labels distribution: {label_counts}")
                for i, count in enumerate(label_counts):
                    class_name = list(self.class_to_label.keys())[i] if i < len(self.class_to_label) else f"class_{i}"
                    print(f"  {class_name}: {count} samples")
            except Exception as e:
                print(f"Error computing label distribution: {e}")
                print(f"Labels sample: {self.labels[:10] if len(self.labels) > 0 else 'Empty'}")
        
        print(f"Class mapping: {self.class_to_label}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        points = self.data[idx]
        if len(points) > self.num_points:
            points = points[:self.num_points]  # Truncate if too many points
        elif len(points) < self.num_points:
            # Pad if too few points (repeat last point)
            padding = np.tile(points[-1:], (self.num_points - len(points), 1))
            points = np.concatenate([points, padding], axis=0)
        
        # Normalize points to zero mean and unit sphere
        points = points - np.mean(points, axis=0)
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / max_dist  # Scale to unit sphere
        
        points = torch.from_numpy(points).float()
        label = torch.tensor(self.labels[idx]).long()
        name = self.names[idx] if idx < len(self.names) else f"sample_{idx}"
        
        return points, label, name

# ==================== Training Functions ====================

def train_vqvae(model, dataloader, optimizer, device, epoch, use_emd=False, emd_weight=1.0):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_embedding_loss = 0
    total_perplexity = 0
    
    for batch_idx, (data, labels, names) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        embedding_loss, x_hat, perplexity = model(data)
        
        # Reconstruction loss - choose between Chamfer Distance and EMD
        if use_emd:
            # Phase 2: Fine-tuning with EMD
            recon_loss = torch.mean(earth_mover_distance(x_hat, data, transpose=False)) * emd_weight
            loss_type = "EMD"
        else:
            # Phase 1: Training with Chamfer Distance
            recon_loss, _ = chamfer_distance(x_hat, data)
            loss_type = "CD"

        # Total loss
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_embedding_loss += embedding_loss.item()
        total_perplexity += perplexity.item()
        
        if batch_idx % 5 == 0:  # Print more frequently for smaller dataset
            print(f'Epoch {epoch}, Batch {batch_idx} ({loss_type}): '
                  f'Loss: {loss.item():.4f}, '
                  f'Recon: {recon_loss.item():.4f}, '
                  f'Embed: {embedding_loss.item():.4f}, '
                  f'Perplexity: {perplexity.item():.2f}')
    
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon_loss / len(dataloader)
    avg_embed = total_embedding_loss / len(dataloader)
    avg_perplexity = total_perplexity / len(dataloader)
    
    return avg_loss, avg_recon, avg_embed, avg_perplexity

def visualize_reconstruction(model, dataloader, device, epoch, num_samples=3):
    model.eval()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    with torch.no_grad():
        data, labels, names = next(iter(dataloader))
        
        # Ensure we don't request more samples than available
        available_samples = data.shape[0]
        actual_samples = min(num_samples, available_samples)
        
        data = data[:actual_samples].to(device)
        
        _, x_hat, _ = model(data)
        
        # Convert to numpy
        original = data.cpu().numpy()
        reconstructed = x_hat.cpu().numpy()
        
        fig = plt.figure(figsize=(15, 5 * actual_samples))
        
        for i in range(actual_samples):
            # Original
            ax1 = fig.add_subplot(actual_samples, 2, 2*i + 1, projection='3d')
            points = original[i]
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.6)
            ax1.set_title(f'Original - {names[i]}')
            
            # Reconstructed
            ax2 = fig.add_subplot(actual_samples, 2, 2*i + 2, projection='3d')
            points_recon = reconstructed[i]
            ax2.scatter(points_recon[:, 0], points_recon[:, 1], points_recon[:, 2], s=1, alpha=0.6)
            ax2.set_title(f'Reconstructed - {names[i]}')
        
        plt.tight_layout()
        
        # Save to logs folder instead of showing
        save_path = f'logs/reconstruction_epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Reconstruction visualization saved to: {save_path}")

# ==================== Main Training Code ====================

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    h_dim = 256
    res_h_dim = 128
    n_res_layers = 3
    n_embeddings = 2048
    embedding_dim = 64
    beta = 0.25
    learning_rate_phase1 = 1e-3  # Learning rate for Phase 1 (Chamfer Distance)
    learning_rate_phase2 = 1e-4  # Lower learning rate for Phase 2 (EMD fine-tuning)
    batch_size = 64  
    num_epochs = 200  
    num_points = 1024
    
    # Two-phase training configuration
    phase1_epochs = int(num_epochs * 0.25)  # 25% for Phase 1 (Chamfer Distance)
    phase2_epochs = num_epochs - phase1_epochs  # 75% for Phase 2 (EMD fine-tuning)
    emd_weight = 0.01  # Weight for EMD loss to prevent it from being too large
    
    # Create dataset and dataloader
    root_dir = "."  # Current directory since filtered_data is in the workspace root
    train_dataset = FilteredDataset(root_dir, num_points=num_points, split='train', max_samples=None)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Set to 0 to avoid Windows multiprocessing issues
    
    # Create model
    model = PointCloudVQVAE(h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta)
    model = model.to(device)
    
    # Initialize optimizer for Phase 1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_phase1)
    scheduler_phase1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    scheduler_phase2 = None  # Will be initialized when switching to Phase 2
    
    # Training loop
    print("Starting Two-Phase Training...")
    print(f"Phase 1 (Chamfer Distance): Epochs 0-{phase1_epochs-1} ({phase1_epochs} epochs - 25%)")
    print(f"Phase 2 (EMD Fine-tuning): Epochs {phase1_epochs}-{num_epochs-1} ({phase2_epochs} epochs - 75%)")
    print("="*60)
    
    train_losses = []
    current_phase = 1
    
    for epoch in range(num_epochs):
        # Check if we need to switch to Phase 2
        if epoch == phase1_epochs:
            print("\n" + "="*60)
            print(f"🔄 SWITCHING TO PHASE 2: EMD Fine-tuning (Epoch {epoch})")
            print("="*60)
            current_phase = 2
            
            # Update optimizer with lower learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate_phase2
            print(f"Learning rate reduced to: {learning_rate_phase2}")
            
            # Initialize scheduler for Phase 2
            scheduler_phase2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            print("Initialized new scheduler for Phase 2")
        
        # Determine training parameters based on phase
        use_emd = (epoch >= phase1_epochs)
        phase_name = "Phase 1 (CD)" if not use_emd else "Phase 2 (EMD)"
        
        # Train the model
        avg_loss, avg_recon, avg_embed, avg_perplexity = train_vqvae(
            model, train_loader, optimizer, device, epoch, 
            use_emd=use_emd, emd_weight=emd_weight
        )
        
        # Step the appropriate scheduler based on phase
        if epoch < phase1_epochs:
            # Phase 1: Use Chamfer Distance scheduler
            scheduler_phase1.step(avg_loss)
        else:
            # Phase 2: Use EMD scheduler
            if scheduler_phase2 is not None:
                scheduler_phase2.step(avg_loss)
        
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch} ({phase_name}): Avg Loss: {avg_loss:.4f}, '
              f'Avg Recon: {avg_recon:.4f}, '
              f'Avg Embed: {avg_embed:.4f}, '
              f'Avg Perplexity: {avg_perplexity:.2f}')
        
        # Visualize reconstruction every 10 epochs
        if epoch % 10 == 0:
            print(f"Visualizing reconstruction ({phase_name})...")
            visualize_reconstruction(model, train_loader, device, epoch)
        
        # Save model checkpoint
        if epoch % 10 == 0 or epoch == phase1_epochs - 1:  # Also save at end of Phase 1
            checkpoint_name = f'vqvae_pointcloud_epoch_{epoch}.pth'
            if epoch == phase1_epochs - 1:
                checkpoint_name = f'vqvae_pointcloud_phase1_final.pth'
            
            torch.save({
                'epoch': epoch,
                'phase': current_phase,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_name)
            
            if epoch == phase1_epochs - 1:
                print(f"📁 Phase 1 final checkpoint saved: {checkpoint_name}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('VQ-VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Save to logs folder
    loss_plot_path = 'logs/training_loss.png'
    plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training loss plot saved to: {loss_plot_path}")
    
    # Final visualization
    print("Final reconstruction visualization...")
    visualize_reconstruction(model, train_loader, device, epoch=num_epochs-1, num_samples=3)
    
    # Save final model
    torch.save(model.state_dict(), 'vqvae_pointcloud_final.pth')
    print("Training completed!")