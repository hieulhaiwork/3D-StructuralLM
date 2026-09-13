import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class BoundingBox:
    def __init__(self, center, edge, label=""):
        self.center = np.array(center)
        self.edge = np.array(edge)
        self.label = label
        self.min_corner = self.center - self.edge / 2
        self.max_corner = self.center + self.edge / 2
        
    def get_volume(self):
        return np.prod(self.edge)
    
    def get_corners(self):
        x_min, y_min, z_min = self.min_corner
        x_max, y_max, z_max = self.max_corner
        return np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
        ])
    
    def intersects(self, other):
        return np.all(self.min_corner <= other.max_corner) and np.all(other.min_corner <= self.max_corner)
    
    def intersection_volume(self, other):
        if not self.intersects(other):
            return 0.0
        intersection_min = np.maximum(self.min_corner, other.min_corner)
        intersection_max = np.minimum(self.max_corner, other.max_corner)
        return np.prod(np.maximum(0, intersection_max - intersection_min))

def draw_box(ax, bbox, color='blue', alpha=0.3):
    corners = bbox.get_corners()
    faces = [
        [corners[0], corners[1], corners[2], corners[3]], [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]], [corners[2], corners[3], corners[7], corners[6]],
        [corners[0], corners[3], corners[7], corners[4]], [corners[1], corners[2], corners[6], corners[5]],
    ]
    face_collection = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidths=1)
    ax.add_collection3d(face_collection)
    ax.text(bbox.center[0], bbox.center[1], bbox.center[2], bbox.label, fontsize=10, weight='bold')

def visualize_bboxes(bboxes, title="3D Bounding Boxes"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, bbox in enumerate(bboxes):
        draw_box(ax, bbox, color=colors[i % len(colors)], alpha=0.3)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    
    all_corners = np.vstack([bbox.get_corners() for bbox in bboxes])
    max_range = np.array([
        all_corners[:, 0].max() - all_corners[:, 0].min(),
        all_corners[:, 1].max() - all_corners[:, 1].min(),
        all_corners[:, 2].max() - all_corners[:, 2].min()
    ]).max() / 2.0
    
    mid = (all_corners.max(axis=0) + all_corners.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.grid(True)
    plt.tight_layout()
    return fig, ax

def load_yaml_and_create_bboxes(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    bboxes = []
    for center, edge, prompt in zip(data.get('center', []), data.get('edge', []), data.get('prompt', [])):
        label = prompt[:27] + "..." if len(prompt) > 30 else prompt
        bboxes.append(BoundingBox(center, edge, label))
    return bboxes, data

def calculate_tor_metric(bboxes):
    if not bboxes or len(bboxes) < 2:
        return 0.0, 0.0, 0.0

    all_min_corners = np.array([b.min_corner for b in bboxes])
    all_max_corners = np.array([b.max_corner for b in bboxes])
    
    scene_min = np.min(all_min_corners, axis=0)
    scene_max = np.max(all_max_corners, axis=0)
    
    scene_volume = np.prod(np.maximum(0, scene_max - scene_min))
    
    if scene_volume == 0:
        return 0.0, 0.0, 0.0

    total_intersection_volume = sum(
        bboxes[i].intersection_volume(bboxes[j]) 
        for i in range(len(bboxes)) 
        for j in range(i + 1, len(bboxes))
    )
    
    tor = total_intersection_volume / scene_volume
    return tor, total_intersection_volume, scene_volume

def main(yaml_file):
    bboxes, data = load_yaml_and_create_bboxes(yaml_file)    
    tor_score, total_inter, scene_vol = calculate_tor_metric(bboxes)

    print(f"Total Intersection Volume: {total_inter:.6f}")
    print(f"Scene Bounding Volume:     {scene_vol:.6f}")
    print(f"TOR Score (Thesis Metric): {tor_score:.6f}")

    scene_name = data.get('scene', 'Scene')
    fig, ax = visualize_bboxes(bboxes, title=f"{scene_name}\nTOR: {tor_score:.4f}")
    
    output_file = yaml_file.replace('.yaml', '_bbox_visualization.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    import sys
    
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "./structuralm_outputs/scene/prompt_15.yaml"
    main(yaml_file)