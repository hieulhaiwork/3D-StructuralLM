# file: src/utils.py

import yaml
import torch
import random
import numpy as np
import os

def load_config(config_path):
    """Tải file cấu hình YAML."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    """Thiết lập seed cho tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def ensure_dir(path):
    """Đảm bảo một thư mục tồn tại."""
    os.makedirs(os.path.dirname(path), exist_ok=True)