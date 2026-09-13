# file: train.py

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from src.utils import load_config, set_seed
from src.dataset import SceneGraphDataset
from src.model import LayoutGAT
from src.trainer import GATTrainer

def main():
    # 1. Tải cấu hình và thiết lập
    config = load_config("configs/configs_syn.yaml")
    set_seed(config['seed'])
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    # 2. Tải và chia Dataset
    # Load temporary dataset để lấy indices cho train/val split
    temp_dataset = SceneGraphDataset(config['data']['processed_path'], split='val')
    train_size = int(len(temp_dataset) * config['data']['train_val_split'])
    val_size = len(temp_dataset) - train_size
    
    # Tạo random split indices
    from torch.utils.data import Subset
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(temp_dataset)), [train_size, val_size]
    )
    
    # Load dataset với augmentation cho train, không augmentation cho val
    train_dataset_full = SceneGraphDataset(config['data']['processed_path'], split='train')
    val_dataset_full = SceneGraphDataset(config['data']['processed_path'], split='val')
    
    # Tạo subsets với đúng indices
    train_dataset = Subset(train_dataset_full, train_indices.indices)
    val_dataset = Subset(val_dataset_full, val_indices.indices)
    
    print(f"\n{'='*60}")
    print(f"Dataset Split: {len(train_dataset)} train / {len(val_dataset)} val")
    print(f"{'='*60}\n")
    
    # 3. Tạo DataLoaders
    # CRITICAL: Phải exclude edge_index từ batching vì PyG tự động xử lý
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        follow_batch=[]  # PyG sẽ tự động xử lý edge_index
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        follow_batch=[]
    )
    
    # 4. Khởi tạo Model và Optimizer
    model = LayoutGAT(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    # 5. Khởi tạo và chạy Trainer
    trainer = GATTrainer(model, train_loader, val_loader, optimizer, config)
    trainer.train()

if __name__ == "__main__":
    # Tạo thư mục checkpoints nếu chưa có
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    main()