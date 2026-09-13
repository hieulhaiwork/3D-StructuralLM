# file: src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from src.mappings import NUM_CLASSES


class LayoutGAT(nn.Module):
    def __init__(self, config):
        super().__init__()

        num_classes = config["model"].get("num_classes", NUM_CLASSES)
        embedding_dim = config["model"].get("embedding_dim", 64)
        bbox_dim = 3

        self.room_scale = config["model"].get("room_scale", 2.0)

        edge_in_dim = config["model"]["edge_in_dim"]
        hidden_dim = config["model"]["hidden_dim"]
        num_heads = config["model"]["num_heads"]
        num_gat_layers = config["model"]["num_gat_layers"]
        dropout_rate = config["model"]["dropout_rate"]

        self.class_embedding = nn.Embedding(num_classes, embedding_dim)

        self.max_nodes_per_scene = 50
        self.index_embedding_dim = 16
        self.index_embedding = nn.Embedding(
            self.max_nodes_per_scene, self.index_embedding_dim
        )

        input_dim = embedding_dim + bbox_dim + self.index_embedding_dim + 1

        self.node_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )

        self.edge_projection = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )

        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        current_dim = hidden_dim

        for i in range(num_gat_layers):
            self.gat_layers.append(
                GATv2Conv(
                    current_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    edge_dim=hidden_dim,
                    add_self_loops=True,
                )
            )

            self.layer_norms.append(nn.LayerNorm(hidden_dim))

            self.residuals.append(nn.Identity())

        self.dropout = nn.Dropout(p=dropout_rate)

        self.translation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4)
        )

    def forward(self, data):
        edge_index, edge_attr = data.edge_index, data.edge_attr

        class_emb = self.class_embedding(data.category)
        global_indices = torch.arange(data.x.size(0), device=data.x.device)
        batch_indices = global_indices % self.max_nodes_per_scene
        index_emb = self.index_embedding(batch_indices)

        is_anchor = torch.zeros((data.x.size(0), 1), device=data.x.device)
        if hasattr(data, "ptr") and len(data.ptr) > 1:
            anchor_indices = data.ptr[:-1]
            is_anchor[anchor_indices] = 1.0
        else:
            is_anchor[0] = 1.0

        x = torch.cat([class_emb, data.x, index_emb, is_anchor], dim=1)
        x = self.node_projection(x)

        edge_attr_emb = self.edge_projection(edge_attr)

        for i in range(len(self.gat_layers)):
            x_in = x

            x = self.gat_layers[i](x, edge_index, edge_attr=edge_attr_emb)

            x = x + self.residuals[i](x_in)
            x = self.layer_norms[i](x)
            x = F.elu(x)
            x = self.dropout(x)

        final_node_features = x

        raw_pos = self.translation_head(final_node_features)
        pred_positions = raw_pos * self.room_scale

        pred_quaternions = self.rotation_head(final_node_features)
        pred_quaternions = F.normalize(pred_quaternions, p=2, dim=-1)

        if hasattr(data, "ptr") and len(data.ptr) > 1:
            pred_positions[data.ptr[:-1]] = 0.0
        else:
            pred_positions[0] = 0.0

        return pred_positions, pred_quaternions
