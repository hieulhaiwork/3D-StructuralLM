import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import ensure_dir


class GATTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config["training"]["device"])

        self.patience = config["training"]["early_stopping_patience"]
        self.min_delta = config["training"]["early_stopping_min_delta"]
        self.checkpoint_path = config["training"]["checkpoint_path"]
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0
        self.early_stop = False

        ensure_dir(self.checkpoint_path)

    def _quaternion_loss(self, pred_quat, true_quat):
        dot_product = torch.sum(pred_quat * true_quat, dim=-1)
        return torch.mean(1.0 - dot_product**2)

    def _compute_bbox_3d(self, centers, bbox_dims):
        half_dims = bbox_dims / 2.0
        bbox_min = centers - half_dims
        bbox_max = centers + half_dims
        return bbox_min, bbox_max

    def _compute_iou_3d(self, bbox1_min, bbox1_max, bbox2_min, bbox2_max):
        inter_min = torch.max(bbox1_min, bbox2_min)
        inter_max = torch.min(bbox1_max, bbox2_max)
        inter_dims = torch.clamp(inter_max - inter_min, min=0.0)
        inter_volume = torch.prod(inter_dims)

        bbox1_dims = bbox1_max - bbox1_min
        bbox2_dims = bbox2_max - bbox2_min
        bbox1_volume = torch.prod(bbox1_dims)
        bbox2_volume = torch.prod(bbox2_dims)

        union_volume = bbox1_volume + bbox2_volume - inter_volume
        iou = inter_volume / (union_volume + 1e-8)
        return iou

    def _collision_loss(self, pred_positions, bbox_dims, batch_ptr):
        loss = torch.tensor(0.0, device=self.device)

        if batch_ptr is None:
            return loss

        num_graphs = len(batch_ptr) - 1
        count = 0

        for i in range(num_graphs):
            start_idx = batch_ptr[i]
            end_idx = batch_ptr[i + 1]

            graph_nodes_pos = pred_positions[start_idx:end_idx]
            graph_nodes_bbox = bbox_dims[start_idx:end_idx]

            num_nodes = graph_nodes_pos.shape[0]
            if num_nodes < 2:
                continue

            mins, maxs = self._compute_bbox_3d(graph_nodes_pos, graph_nodes_bbox)

            for n1 in range(num_nodes):
                for n2 in range(n1 + 1, num_nodes):
                    iou = self._compute_iou_3d(mins[n1], maxs[n1], mins[n2], maxs[n2])
                    loss += iou
                    count += 1

        if count > 0:
            loss = loss / count
        return loss

    def _geometric_consistency_loss(self, pred_pos, edge_index, edge_attr):
        if edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=pred_pos.device)

        src_pos = pred_pos[edge_index[0]]
        dst_pos = pred_pos[edge_index[1]]

        loss = 0.0

        MARGIN = 0.05

        MARGIN_Y_ABOVE = 0.5

        is_right = edge_attr[:, 4] > 0.5
        loss += torch.sum(torch.relu(dst_pos[:, 2] + MARGIN - src_pos[:, 2]) * is_right)

        is_left = edge_attr[:, 3] > 0.5
        loss += torch.sum(
            torch.relu(src_pos[:, 2] - (dst_pos[:, 2] - MARGIN)) * is_left
        )

        is_front = edge_attr[:, 5] > 0.5
        loss += torch.sum(torch.relu(dst_pos[:, 0] + MARGIN - src_pos[:, 0]) * is_front)

        is_behind = edge_attr[:, 6] > 0.5
        loss += torch.sum(
            torch.relu(src_pos[:, 0] - (dst_pos[:, 0] - MARGIN)) * is_behind
        )

        is_above = (edge_attr[:, 1] > 0.5) | (edge_attr[:, 0] > 0.5)
        loss += torch.sum(
            torch.relu(dst_pos[:, 1] + MARGIN_Y_ABOVE - src_pos[:, 1]) * is_above
        )

        is_under = edge_attr[:, 2] > 0.5
        loss += torch.sum(torch.relu(src_pos[:, 1] - (dst_pos[:, 1] - 0.01)) * is_under)

        return loss

    def _combined_loss(
        self,
        pred_pos,
        pred_rot,
        true_pos,
        true_rot,
        bbox_dims,
        edge_index,
        edge_attr,
        batch_ptr,
    ):
        N = pred_pos.shape[0]

        anchor_mask = torch.ones(N, device=self.device)

        if batch_ptr is not None:
            anchor_indices = batch_ptr[:-1]
            anchor_mask[anchor_indices] = 0.0
        else:
            anchor_mask[0] = 0.0

        pos_diff = torch.abs(pred_pos - true_pos)
        pos_loss = (pos_diff * anchor_mask.unsqueeze(1)).sum() / (
            anchor_mask.sum() * 3 + 1e-8
        )

        dot_product = torch.sum(pred_rot * true_rot, dim=-1)
        rot_loss_per_node = 1.0 - dot_product**2
        rot_loss = (rot_loss_per_node * anchor_mask).sum() / (anchor_mask.sum() + 1e-8)

        collision_loss = self._collision_loss(pred_pos, bbox_dims, batch_ptr)

        geometric_loss = self._geometric_consistency_loss(
            pred_pos, edge_index, edge_attr
        )

        total_loss = (
            (self.config["training"]["pos_loss_weight"] * pos_loss)
            + (self.config["training"]["rot_loss_weight"] * rot_loss)
            + (self.config["training"]["collision_loss_weight"] * collision_loss)
            + (self.config["training"]["geometric_loss_weight"] * geometric_loss)
        )

        return total_loss, pos_loss, rot_loss, collision_loss, geometric_loss

    def _train_epoch(self):
        self.model.train()
        total_loss, total_pos, total_rot, total_coll, total_geo = 0, 0, 0, 0, 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            pred_pos, pred_rot = self.model(batch)
            bbox_dims = batch.x

            loss, pos_l, rot_l, coll_l, geo_l = self._combined_loss(
                pred_pos,
                pred_rot,
                batch.y_pos,
                batch.y_rot,
                bbox_dims,
                batch.edge_index,
                batch.edge_attr,
                batch.ptr,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_pos += pos_l.item()
            total_rot += rot_l.item()
            total_coll += coll_l.item()
            total_geo += geo_l.item()

        return (
            total_loss / len(self.train_loader),
            total_pos / len(self.train_loader),
            total_rot / len(self.train_loader),
            total_coll / len(self.train_loader),
            total_geo / len(self.train_loader),
        )

    def _validate_epoch(self):
        self.model.eval()
        total_loss, total_pos, total_rot, total_coll, total_geo = 0, 0, 0, 0, 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                batch = batch.to(self.device)
                pred_pos, pred_rot = self.model(batch)
                bbox_dims = batch.x

                loss, pos_l, rot_l, coll_l, geo_l = self._combined_loss(
                    pred_pos,
                    pred_rot,
                    batch.y_pos,
                    batch.y_rot,
                    bbox_dims,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.ptr,
                )

                total_loss += loss.item()
                total_pos += pos_l.item()
                total_rot += rot_l.item()
                total_coll += coll_l.item()
                total_geo += geo_l.item()

        val_loss = total_loss / len(self.val_loader)

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            print(f"Validation loss improved. Saving model to {self.checkpoint_path}")
            torch.save(self.model.state_dict(), self.checkpoint_path)
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.patience:
                self.early_stop = True

        return (
            val_loss,
            total_pos / len(self.val_loader),
            total_rot / len(self.val_loader),
            total_coll / len(self.val_loader),
            total_geo / len(self.val_loader),
        )

    def train(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )

        print(f"Starting training on {self.device}...")
        print(f"Initial LR: {self.optimizer.param_groups[0]['lr']}")

        for epoch in range(self.config["training"]["epochs"]):
            train_metrics = self._train_epoch()
            val_metrics = self._validate_epoch()

            t_loss, t_pos, t_rot, t_coll, t_geo = train_metrics
            v_loss, v_pos, v_rot, v_coll, v_geo = val_metrics

            scheduler.step(v_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} | LR: {current_lr:.6f}"
            )
            print(
                f"  Train: Loss={t_loss:.4f} (Pos={t_pos:.3f}, Rot={t_rot:.3f}, Coll={t_coll:.3f}, Geo={t_geo:.3f})"
            )
            print(
                f"  Val:   Loss={v_loss:.4f} (Pos={v_pos:.3f}, Rot={v_rot:.3f}, Coll={v_coll:.3f}, Geo={v_geo:.3f})"
            )
            print("-" * 60)

            if self.early_stop:
                print("Early stopping triggered!")
                break

        print("Training completed.")
