# file: src/dataset.py

import torch
from torch_geometric.data import Dataset

class SceneGraphDataset(Dataset):
    """
    Lớp Dataset để tải dữ liệu đồ thị cảnh đã được tiền xử lý.
    Dataset sử dụng learned embeddings thay vì SentenceTransformer.
    
    QUAN TRỌNG: Phải kế thừa từ torch_geometric.data.Dataset
    để DataLoader tự động xử lý batching của graphs đúng cách!
    
    Augmentation:
    - Scale: 0.85-1.15x (±15% size variation)
    - Flip Z: Mirror scene qua mặt phẳng XY (swap left_of ↔ right_of)
    """
    def __init__(self, processed_data_path, split='train', transform=None, pre_transform=None):
        """
        Args:
            split (str): 'train' hoặc 'val'. Augmentation chỉ áp dụng khi split='train'.
        """
        self.processed_data_path = processed_data_path
        self.split = split
        try:
            self.data_list = torch.load(processed_data_path, weights_only=False)
            aug_status = "(Augmentation ON)" if split == 'train' else "(Augmentation OFF)"
            print(f"[{split.upper()}] Đã tải {len(self.data_list)} đồ thị từ {processed_data_path} {aug_status}")
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file {processed_data_path}.")
            print("Hãy đảm bảo bạn đã chạy script tiền xử lý để tạo file này.")
            exit()
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        # Clone để không sửa dữ liệu gốc trong memory
        data = self.data_list[idx].clone()

        if self.transform is not None:
            data = self.transform(data)

        # Chỉ augment khi training
        if self.split == 'train':
            data = self._augment(data)
        
        return data
    
    def _augment(self, data):
        """
        Augmentation an toàn cho anchor-based layout với semantic alignment.
        
        1. Random Scale: Co dãn toàn bộ scene (85%-115%)
        2. Random Flip Z: Lật gương qua mặt phẳng XY + swap left/right relationships
        """
        # --- 1. RANDOM SCALE (85% - 115%) ---
        scale = torch.rand(1).item() * 0.30 + 0.85  # [0.85, 1.15]
        
        # Scale bbox (data.x chỉ chứa bbox 3D)
        data.x = data.x * scale
        
        # Scale positions (giữ anchor tại origin vì scale tương đối)
        data.y_pos = data.y_pos * scale
        
        # --- 2. RANDOM HORIZONTAL FLIP (50% xác suất) ---
        if torch.rand(1).item() > 0.5:
            # A. Lật tọa độ Z: z → -z (mirror qua mặt phẳng XY)
            data.y_pos[:, 2] = -data.y_pos[:, 2]
            
            # B. Lật quaternion rotation
            # Khi mirror Z-axis: [w, x, y, z] → [w, x, -y, z]
            # (Đảo chiều rotation quanh Y-axis để giữ hướng đối tượng nhất quán)
            data.y_rot[:, 2] = -data.y_rot[:, 2]
            
            # C. Swap relationship labels: left_of ↔ right_of
            # Edge attr: [E, 8] với indices:
            # 0=front_of, 1=behind, 2=on_top_of, 3=left_of, 4=right_of, 5=close_by, 6=same_as, 7=symmetrical_to
            left_col = data.edge_attr[:, 3].clone()
            right_col = data.edge_attr[:, 4].clone()
            data.edge_attr[:, 3] = right_col
            data.edge_attr[:, 4] = left_col
        
        return data