# file: src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from src.mappings import NUM_CLASSES

class LayoutGAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Config
        num_classes = config['model'].get('num_classes', NUM_CLASSES)
        embedding_dim = config['model'].get('embedding_dim', 64)
        bbox_dim = 3  # width, height, depth

        # Scale output cho Tanh (lấy từ config, mặc định 2.0)
        self.room_scale = config['model'].get('room_scale', 2.0)

        edge_in_dim = config['model']['edge_in_dim']
        hidden_dim = config['model']['hidden_dim']
        num_heads = config['model']['num_heads']
        num_gat_layers = config['model']['num_gat_layers']
        dropout_rate = config['model']['dropout_rate']
        
        # --- EMBEDDINGS ---
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        # --- INDEX EMBEDDING (Symmetry Breaking) ---
        self.max_nodes_per_scene = 50  # Max objects trong 1 scene
        self.index_embedding_dim = 16
        self.index_embedding = nn.Embedding(self.max_nodes_per_scene, self.index_embedding_dim)
        
        # Input: [Class(64) + Bbox(3) + Index(16) + IsAnchor(1)]
        input_dim = embedding_dim + bbox_dim + self.index_embedding_dim + 1
        
        # --- PROJECTIONS ---
        self.node_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Edge MLP: Quan trọng để hiểu các quan hệ phức tạp ở Phase sau
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # --- BACKBONE: DEEP GATv2 ---
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        current_dim = hidden_dim

        for i in range(num_gat_layers):
            self.gat_layers.append(
                GATv2Conv(
                    current_dim, 
                    hidden_dim // num_heads, # Output per head
                    heads=num_heads, 
                    concat=True, 
                    edge_dim=hidden_dim, # Edge features cũng được update
                    add_self_loops=True
                )
            )

            # Norm sau mỗi layer giúp train mạng sâu (Deep GNN) cho Phase 3, 4
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

            # Residual connection (nếu dim đổi thì cần project, ở đây giữ nguyên dim)
            self.residuals.append(nn.Identity())

        # --- OUTPUT HEADS ---
        self.dropout = nn.Dropout(p=dropout_rate)

        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh() # Scaled Tanh
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4) # Output quaternion
        )

    def forward(self, data):
        # data.x: bbox dimensions (N, 3)
        # data.category: class IDs (N,)
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        # 1. Prepare Node Features
        class_emb = self.class_embedding(data.category)  # (N, embedding_dim)
        
        # Index Embedding (Symmetry Breaking)
        global_indices = torch.arange(data.x.size(0), device=data.x.device)
        batch_indices = global_indices % self.max_nodes_per_scene
        index_emb = self.index_embedding(batch_indices)  # (N, index_embedding_dim)
        
        # Mark which nodes are anchors (node 0 of each graph in batch)
        is_anchor = torch.zeros((data.x.size(0), 1), device=data.x.device)
        if hasattr(data, 'ptr') and len(data.ptr) > 1:
            # data.ptr: [0, num_nodes_g1, num_nodes_g1+g2, ..., total]
            anchor_indices = data.ptr[:-1]  # First node of each graph
            is_anchor[anchor_indices] = 1.0
        else:
            # Fallback for batch_size=1
            is_anchor[0] = 1.0
        
        # Concatenate [Class Embedding + Bbox + Index Embedding + IsAnchor]
        x = torch.cat([class_emb, data.x, index_emb, is_anchor], dim=1)  # (N, 84)
        x = self.node_projection(x)
        
        # 2. Prepare Edge Features
        edge_attr_emb = self.edge_projection(edge_attr)
        
        # 3. GAT Backbone Layers
        for i in range(len(self.gat_layers)):
            x_in = x
            
            # Message Passing
            x = self.gat_layers[i](x, edge_index, edge_attr=edge_attr_emb)
            
            # Residual + Norm + Activation
            x = x + self.residuals[i](x_in)
            x = self.layer_norms[i](x)
            x = F.elu(x)
            x = self.dropout(x)


        final_node_features = x
        
        # 4. Heads
        # Position prediction với SCALED TANH
        raw_pos = self.translation_head(final_node_features)  # Output: [-1, 1]
        pred_positions = raw_pos * self.room_scale  # Scale to [-2.5, 2.5]
        
        # Rotation prediction
        pred_quaternions = self.rotation_head(final_node_features)
        pred_quaternions = F.normalize(pred_quaternions, p=2, dim=-1)
        
        # 5. Hard Constraint for Anchor
        # Ép Anchor về (0,0,0) để đảm bảo tính nhất quán cho các Phase sau
        if hasattr(data, 'ptr') and len(data.ptr) > 1:
            pred_positions[data.ptr[:-1]] = 0.0  # Set all anchors to origin
        else:
            pred_positions[0] = 0.0  # Fallback for single graph

        return pred_positions, pred_quaternions