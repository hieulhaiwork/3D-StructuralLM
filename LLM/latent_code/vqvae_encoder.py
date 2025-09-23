# vaqvae_encoder.py
"""VQ-VAE Model Architecture for 3D Point Clouds"""

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
        self.conv_stack = nn.Sequential(
            nn.Conv1d(3, h_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim // 2, h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        # x shape: (batch, 3, num_points)
        return self.conv_stack(x)

class PointCloudDecoder(nn.Module):
    def __init__(self, embedding_dim, h_dim, n_res_layers, res_h_dim):
        super(PointCloudDecoder, self).__init__()
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose1d(h_dim, h_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(h_dim // 2, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
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

class VQVAEEncoderLoader:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"Model encoder loaded successfully from {model_path}.")

    def _load_model(self, model_path: str) -> PointCloudVQVAE:
        """
        Tải mô hình VQ-VAE từ file .pth.
        """
        # 1. Khởi tạo mô hình với các siêu tham số lấy từ file train
        h_dim = 128
        res_h_dim = 64
        n_res_layers = 3
        n_embeddings = 512
        embedding_dim = 64
        beta = 0.25

        model = PointCloudVQVAE(
            h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim, beta
        )

        # 2. Tải trọng số
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ VQ-VAE Encoder loaded successfully from '{model_path}' (epoch {checkpoint.get('epoch', 'N/A')}).")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

        return model.to(self.device)
    
    def encode(self, point_cloud_np: np.ndarray) -> np.ndarray:
        """
        Mã hóa một point cloud (dạng NumPy) thành một chuỗi các mã latent (dạng NumPy).
        """
        with torch.no_grad():
            # 1. Chuyển NumPy array sang PyTorch tensor
            point_cloud_tensor = torch.from_numpy(point_cloud_np).float().to(self.device)
            
            # 2. Thêm batch dimension nếu cần
            if point_cloud_tensor.dim() == 2:
                point_cloud_tensor = point_cloud_tensor.unsqueeze(0) # Shape: (1, num_points, 3)

            # 3. Đưa qua mô hình để lấy mã
            #    x shape: (batch, num_points, 3) -> (batch, 3, num_points)
            x = point_cloud_tensor.permute(0, 2, 1)

            # Encode
            z_e = self.model.encoder(x)
            z_e = self.model.pre_quantization_conv(z_e)
            
            # Lấy mã (indices) từ lớp VectorQuantizer
            z_e_permuted = z_e.permute(0, 2, 1).contiguous()
            z_flattened = z_e_permuted.view(-1, self.model.vector_quantization.e_dim)

            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.model.vector_quantization.embedding.weight**2, dim=1) - 2 * \
                torch.matmul(z_flattened, self.model.vector_quantization.embedding.weight.t())

            min_encoding_indices = torch.argmin(d, dim=1)
            
            # 4. Reshape và chuyển về NumPy
            num_points = point_cloud_tensor.shape[1]
            latent_codes = min_encoding_indices.view(-1, num_points) # Shape: (1, num_points)
            
            # Trả về mảng 1D
            return latent_codes.squeeze().cpu().numpy()