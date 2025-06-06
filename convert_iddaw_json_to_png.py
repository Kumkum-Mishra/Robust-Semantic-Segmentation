import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.draw import polygon as sk_polygon

LABEL_NAME_TO_ID = {
    'road': 0, 'sidewalk': 1, 'building': 2, 'wall': 3, 'fence': 4,
    'pole': 5, 'traffic light': 6, 'traffic sign': 7, 'vegetation': 8,
    'terrain': 9, 'sky': 10, 'person': 11, 'rider': 12, 'car': 13,
    'truck': 14, 'bus': 15, 'train': 16, 'motorcycle': 17, 'bicycle': 18,
    'void': 255
}

def draw_json_to_mask(json_path, save_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    height, width = data.get("imgHeight"), data.get("imgWidth")
    if not height or not width:
        print(f"Invalid size in: {json_path}")
        return

    mask = np.zeros((height, width), dtype=np.uint8)
    drawn = 0

    for obj in data.get("objects", []):
        label = obj.get("label", "").strip()
        polygon = obj.get("polygon", [])

        if label not in LABEL_NAME_TO_ID:
            continue
        if len(polygon) < 3:
            continue

        try:
            poly_np = np.array(polygon)
            rr, cc = sk_polygon(poly_np[:, 1], poly_np[:, 0], mask.shape)
            mask[rr, cc] = LABEL_NAME_TO_ID[label]
            drawn += 1
        except Exception as e:
            print(f"Failed polygon: {label} in {json_path} — {e}")

    if drawn == 0:
        print(f"⚠️ No valid polygons in {json_path}")

    Image.fromarray(mask).save(save_path)

def convert_all(base_dir="data/IDD/IDDAW/train"):
    weather_folders = ["FOG", "RAIN", "LOWLIGHT", "SNOW"]
    for weather in weather_folders:
        input_dir = os.path.join(base_dir, weather, "gtSeg")
        output_dir = os.path.join(base_dir, weather, "gtSegPNG")
        print(f"\nProcessing {weather}...")

        json_files = []
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                if f.endswith(".json"):
                    json_files.append(os.path.join(root, f))

        if not json_files:
            print(f"No JSONs found in {input_dir}")
            continue

        for json_path in tqdm(json_files, desc=f"Converting {weather}"):
            rel_path = os.path.relpath(json_path, input_dir)
            png_path = os.path.join(output_dir, rel_path).replace(".json", ".png")
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            draw_json_to_mask(json_path, png_path)


convert_all()
