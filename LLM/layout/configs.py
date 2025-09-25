from pathlib import Path


# ----------PATHS----------
BASE_DATASET_PATH = Path("./")
# Test
try:
    all_full_glb_files = BASE_DATASET_PATH.rglob("*_full.glb")

    SCENE_FILE_PATHS = [
        p.as_posix() 
        for p in all_full_glb_files 
        if "bedroom" in p.name.lower()
    ]
    
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy thư mục dataset tại '{BASE_DATASET_PATH}'. Vui lòng kiểm tra lại đường dẫn.")
    SCENE_FILE_PATHS = []

OUTPUT_FILE_PATH = Path("./scenegraph_dataset_for_finetuning_v2.json")

# ---------PROCESSING CONFIGS----------
# Objects to be detected
OBJECT_CLASSES = ["Bed", "Cabinet", "Table"]

NUM_POINTS_PER_OBJECT = 2048

MIN_VERTICES_THRESHOLD = 100

# ---------RELATIONSHIP CONFIGS----------
# Ngưỡng (mét) để xác định một vật thể đang đứng trên sàn
FLOOR_Z_THRESHOLD = 0.05
# Ngưỡng khoảng cách (mét) để xác định hai vật thể "cạnh nhau"
NEXT_TO_DISTANCE_THRESHOLD = 2.0
# Ngưỡng (radian) để xác định hai vật thể "thẳng hàng"
ALIGNMENT_AXIS_THRESHOLD = 0.1

# ---------SHAPE CONFIGS----------
TABLE_HEIGHT_THRESHOLD = 0.5  # mét

