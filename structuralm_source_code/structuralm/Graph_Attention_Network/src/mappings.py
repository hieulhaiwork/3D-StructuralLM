"""
Mapping definitions for object classes and spatial relationships.
"""

CLASS_MAPPING = {
    "Armchair": 0,
    "Banana": 1,
    "Basket": 2,
    "Bed": 3,
    "Cabinet": 4,
    "Cake": 5,
    "Carpet": 6,
    "Chair": 7,
    "Cup": 8,
    "Easel": 9,
    "Fan": 10,
    "Fridge": 11,
    "Floor_lamp": 12,
    "Laptop_close": 13,
    "Laptop_open": 14,
    "Lighting": 15,
    "Mirror": 16,
    "Monkey": 17,
    "Panda": 18,
    "Piano": 19,
    "Picture": 20,
    "Pier": 21,
    "Pillow": 22,
    "Rabbit": 23,
    "Sofa": 24,
    "TV": 25,
    "Table": 26,
    "Test": 27,
    "Toy": 28,
    "Tree": 29,
    "Vase": 30,
    "Wardrobe": 31,
    "Washingmachine": 32,
    "human_lying": 33,
    "human_sitting": 34,
    "human_standing": 35,
    "Other": 36,
}

NUM_CLASSES = len(CLASS_MAPPING)

RELATION_MAPPING = {
    "on_top_of": 0,
    "above": 1,
    "under": 2,
    "left_of": 3,
    "right_of": 4,
    "in_front_of": 5,
    "behind": 6,
    "facing": 7,
    "in": 8,
}

NUM_RELATIONS = len(RELATION_MAPPING)

REVERSE_RELATION_MAPPING = {
    "left_of": "right_of",
    "right_of": "left_of",
    "above": "under",
    "under": "above",
    "in_front_of": "behind",
    "behind": "in_front_of",
    "on_top_of": "under",
    "facing": "facing",
    "in": "surrounding",
}


def get_class_id(label):
    """
    Lấy class ID từ label string. Nếu không tìm thấy, trả về ID của "Other"
    """
    return CLASS_MAPPING.get(label, CLASS_MAPPING["Other"])


def get_relation_id(relation):
    """
    Lấy relation ID từ relation string. Trả về None nếu không tìm thấy
    """
    return RELATION_MAPPING.get(relation)


def get_reverse_relation(relation):
    """
    Lấy tên quan hệ ngược. Trả về None nếu không có
    """
    return REVERSE_RELATION_MAPPING.get(relation)
