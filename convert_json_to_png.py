import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

IS_BINARY_MASK = False  # False for IDD Segmentation

LABEL_NAME_TO_ID = {
    'road': 0,
    'sidewalk': 1,
    'building': 2,
    'wall': 3,
    'fence': 4,
    'pole': 5,
    'traffic light': 6,
    'traffic sign': 7,
    'vegetation': 8,
    'terrain': 9,
    'sky': 10,
    'person': 11,
    'rider': 12,
    'car': 13,
    'truck': 14,
    'bus': 15,
    'train': 16,
    'motorcycle': 17,
    'bicycle': 18,
    'void': 255
}

def draw_mask_from_json(json_path, out_path, binary_mask=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    height = data.get("imgHeight")
    width = data.get("imgWidth")
    if not height or not width:
        print(f"Skipping {json_path}, missing height/width")
        return

    mask = np.zeros((height, width), dtype=np.uint8)
    img_pil = Image.fromarray(mask)
    draw = ImageDraw.Draw(img_pil)

    for obj in data.get("objects", []):
        label = obj.get("label", "")
        polygon = obj.get("polygon", [])
        
        if not polygon or len(polygon) < 3:
            continue

        try:
            polygon_int = [tuple(map(int, map(round, point))) for point in polygon]
        except Exception as e:
            print(f"Polygon conversion failed in {json_path}: {e}")
            continue

        fill_value = 1 if binary_mask else LABEL_NAME_TO_ID.get(label, 255)

        try:
            draw.polygon(polygon_int, fill=fill_value)
        except Exception as e:
            print(f"Failed to draw polygon in {json_path}: {e}")
            continue

    img_pil.save(out_path)

def convert_all_jsons(input_root, output_root, binary_mask=False):
    os.makedirs(output_root, exist_ok=True)

    for subfolder in tqdm(os.listdir(input_root), desc="Processing Folders"):
        in_path = os.path.join(input_root, subfolder)
        if not os.path.isdir(in_path):
            continue

        out_path = os.path.join(output_root, subfolder)
        os.makedirs(out_path, exist_ok=True)

        for file in os.listdir(in_path):
            if not file.endswith(".json"):
                continue

            json_file = os.path.join(in_path, file)
            png_file = os.path.join(out_path, file.replace(".json", ".png"))

            draw_mask_from_json(json_file, png_file, binary_mask=binary_mask)

# === RUN HERE ===

convert_all_jsons(
    "data/IDD/IDD_Segmentation/gtFine/train",
    "data/IDD/IDD_Segmentation/gtFinePNG/train",
    binary_mask=False
)
# Convert val set
convert_all_jsons(
    "data/IDD/IDD_Segmentation/gtFine/val",
    "data/IDD/IDD_Segmentation/gtFinePNG/val",
    binary_mask=False
)