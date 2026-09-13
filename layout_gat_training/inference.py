# file: inference.py

"""
Script để triển khai model GAT Layout Prediction với input từ file scene.json
- Load scene.json chứa thông tin node và edge
- Chuyển đổi sang format PyTorch Geometric Data
- Predict position và rotation cho các objects
- Export kết quả ra file JSON
"""

import torch
import json
import numpy as np
from torch_geometric.data import Data
import argparse
import os
from pathlib import Path

from src.utils import load_config
from src.model import LayoutGAT
from src.mappings import CLASS_MAPPING, RELATION_MAPPING, get_reverse_relation


class SceneInference:
    """Class để thực hiện inference trên scene từ file JSON"""
    
    def __init__(self, checkpoint_path, config_path='configs/configs_syn.yaml'):
        """
        Args:
            checkpoint_path: Đường dẫn đến model checkpoint (.pt)
            config_path: Đường dẫn đến file config
        """
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = LayoutGAT(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state_dict format
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
        
        # Reverse mapping
        self.class_labels = {v: k for k, v in CLASS_MAPPING.items()}
    
    def load_scene(self, scene_json_path):
        """
        Load scene từ file JSON
        
        Định dạng JSON mong đợi:
        {
            "objects": [
                {
                    "id": 0,
                    "label": "Bed",
                    "normalized_bounding_box": [width, height, depth],
                    "normalized_relative_center": [x, y, z],  # Optional (dùng cho anchor)
                    "rot": [w, x, y, z]  # Optional quaternion
                }
            ],
            "relationships": [
                {
                    "obj_id1": 0,
                    "obj_id2": 1,
                    "relation": "left_of"
                }
            ]
        }
        
        Returns:
            dict: Scene data
        """
        with open(scene_json_path, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
        
        print(f"Loaded scene with {len(scene_data['objects'])} objects "
              f"and {len(scene_data['relationships'])} relationships")
        return scene_data
    
    def scene_to_graph(self, scene_data):
        """
        Chuyển đổi scene data sang PyTorch Geometric Data
        
        Args:
            scene_data: Dict chứa objects và relationships
            
        Returns:
            torch_geometric.data.Data: Graph data
        """
        objects = scene_data['objects']
        relationships = scene_data.get('relationships', [])
        
        num_nodes = len(objects)
        
        # --- NODE FEATURES ---
        # x: Bounding box dimensions (N, 3)
        bbox_list = []
        category_list = []
        object_names = []
        
        for obj in objects:
            # Bbox dimensions
            bbox = obj['normalized_bounding_box']
            bbox_list.append(bbox)
            
            # Category
            label = obj['label']
            cat_id = CLASS_MAPPING.get(label, CLASS_MAPPING['Other'])
            category_list.append(cat_id)
            
            # Store name for output
            object_names.append(obj.get('name', f"{label}_{obj['id']}"))
        
        x = torch.tensor(bbox_list, dtype=torch.float32)
        category = torch.tensor(category_list, dtype=torch.long)
        
        # --- EDGE INDEX & EDGE ATTR ---
        edge_index_list = []
        edge_attr_list = []
        
        # Tạo mapping từ obj_id sang node index
        id_to_idx = {obj['id']: idx for idx, obj in enumerate(objects)}
        
        for rel in relationships:
            src_id = rel['obj_id1']
            dst_id = rel['obj_id2']
            relation = rel['relation']
            
            # Check valid ids
            if src_id not in id_to_idx or dst_id not in id_to_idx:
                print(f"Warning: Skipping edge with invalid object id: {src_id} -> {dst_id}")
                continue
            
            src_idx = id_to_idx[src_id]
            dst_idx = id_to_idx[dst_id]
            
            # Get relation ID
            rel_id = RELATION_MAPPING.get(relation, -1)
            if rel_id == -1:
                print(f"Warning: Unknown relation '{relation}', skipping edge")
                continue
            
            # Create one-hot encoding for relation (9 dimensions)
            edge_attr = [0.0] * 9
            edge_attr[rel_id] = 1.0
            
            # Add forward edge
            edge_index_list.append([src_idx, dst_idx])
            edge_attr_list.append(edge_attr)
            
            # Add reverse edge with reverse relation (matching training data)
            reverse_relation = get_reverse_relation(relation)
            if reverse_relation:
                rev_rel_id = RELATION_MAPPING.get(reverse_relation, -1)
                if rev_rel_id != -1:
                    rev_edge_attr = [0.0] * 9
                    rev_edge_attr[rev_rel_id] = 1.0
                    edge_index_list.append([dst_idx, src_idx])
                    edge_attr_list.append(rev_edge_attr)
            else:
                # Nếu không có reverse relation, dùng same relation (như "facing")
                edge_index_list.append([dst_idx, src_idx])
                edge_attr_list.append(edge_attr)
        
        if len(edge_index_list) == 0:
            # Nếu không có edge, tạo self-loop cho node 0
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0.0] * 9], dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        
        # --- CREATE DATA OBJECT ---
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            category=category
        )
        
        # Store metadata
        data.object_names = object_names
        data.num_nodes = num_nodes
        
        return data
    
    def predict(self, data):
        """
        Thực hiện inference trên graph data
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            tuple: (predicted_positions, predicted_rotations)
                - predicted_positions: (N, 3) numpy array
                - predicted_rotations: (N, 4) numpy array (quaternions)
        """
        with torch.no_grad():
            data = data.to(self.device)
            pred_positions, pred_rotations = self.model(data)
            
            # Convert to numpy
            pred_positions = pred_positions.cpu().numpy()
            pred_rotations = pred_rotations.cpu().numpy()
        
        return pred_positions, pred_rotations
    
    def save_results(self, scene_data, predictions, output_path):
        """
        Lưu kết quả prediction ra file JSON
        
        Args:
            scene_data: Original scene data
            predictions: Tuple (positions, rotations)
            output_path: Đường dẫn file output
        """
        # Create output directory if not exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        positions, rotations = predictions
        
        # Create output structure
        output = {
            "metadata": {
                "model": "LayoutGAT",
                "num_objects": len(scene_data['objects']),
                "device": str(self.device)
            },
            "predictions": []
        }
        
        for idx, obj in enumerate(scene_data['objects']):
            pred_obj = {
                "id": obj['id'],
                "name": obj.get('name', f"{obj['label']}_{obj['id']}"),
                "label": obj['label'],
                "input": {
                    "bounding_box": obj['normalized_bounding_box'],
                },
                "prediction": {
                    "position": positions[idx].tolist(),
                    "rotation_quaternion": rotations[idx].tolist(),
                    "rotation_format": "xyzw"
                }
            }
            
            # Add original position/rotation if available
            if 'normalized_relative_center' in obj:
                pred_obj['input']['position'] = obj['normalized_relative_center']
            if 'rot' in obj:
                pred_obj['input']['rotation'] = obj['rot']
            
            output['predictions'].append(pred_obj)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
    
    def run_inference(self, scene_json_path, output_path=None):
        """
        Pipeline đầy đủ: load scene -> predict -> save results
        
        Args:
            scene_json_path: Đường dẫn file scene.json
            output_path: Đường dẫn file output (mặc định: scene_json_path + '_predictions.json')
        """
        # Load scene
        scene_data = self.load_scene(scene_json_path)
        
        # Convert to graph
        print("Converting scene to graph...")
        graph_data = self.scene_to_graph(scene_data)
        print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges")
        
        # Predict
        print("Running inference...")
        predictions = self.predict(graph_data)
        
        # Print summary
        positions, rotations = predictions
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        for idx, obj in enumerate(scene_data['objects']):
            print(f"\nObject {idx}: {obj['label']}")
            print(f"  Position: [{positions[idx][0]:.4f}, {positions[idx][1]:.4f}, {positions[idx][2]:.4f}]")
            print(f"  Rotation: [{rotations[idx][0]:.4f}, {rotations[idx][1]:.4f}, "
                  f"{rotations[idx][2]:.4f}, {rotations[idx][3]:.4f}]")
        print("="*60)
        
        # Save results
        if output_path is None:
            output_path = str(Path(scene_json_path).with_suffix('')) + '_predictions.json'
        
        self.save_results(scene_data, predictions, output_path)
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description='GAT Layout Model Inference')
    parser.add_argument('--scene', type=str, required=True,
                        help='Path to scene.json file')
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/model_phase1234.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                        default='configs/configs_syn.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output JSON file (default: <scene>_predictions.json)')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.scene):
        print(f"Error: Scene file not found: {args.scene}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Run inference
    inferencer = SceneInference(args.checkpoint, args.config)
    inferencer.run_inference(args.scene, args.output)


if __name__ == '__main__':
    main()
