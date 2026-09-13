# file: src/dataset.py

import torch
from torch_geometric.data import Dataset


class SceneGraphDataset(Dataset):
    def __init__(
        self, processed_data_path, split="train", transform=None, pre_transform=None
    ):
        self.processed_data_path = processed_data_path
        self.split = split
        try:
            self.data_list = torch.load(processed_data_path, weights_only=False)
            aug_status = (
                "(Augmentation ON)" if split == "train" else "(Augmentation OFF)"
            )
            print(
                f"[{split.upper()}] Đã tải {len(self.data_list)} đồ thị từ {processed_data_path} {aug_status}"
            )
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file {processed_data_path}.")
            print("Hãy đảm bảo bạn đã chạy script tiền xử lý để tạo file này.")
            exit()
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx].clone()

        if self.transform is not None:
            data = self.transform(data)

        if self.split == "train":
            data = self._augment(data)

        return data

    def _augment(self, data):
        scale = torch.rand(1).item() * 0.30 + 0.85

        data.x = data.x * scale

        data.y_pos = data.y_pos * scale

        if torch.rand(1).item() > 0.5:
            data.y_pos[:, 2] = -data.y_pos[:, 2]

            data.y_rot[:, 2] = -data.y_rot[:, 2]

            left_col = data.edge_attr[:, 3].clone()
            right_col = data.edge_attr[:, 4].clone()
            data.edge_attr[:, 3] = right_col
            data.edge_attr[:, 4] = left_col

        return data
