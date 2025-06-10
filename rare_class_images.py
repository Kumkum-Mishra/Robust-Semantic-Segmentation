import os
import numpy as np
from PIL import Image
from collections import defaultdict

# Path to training masks
mask_root = "data/IDD/IDD_Segmentation/gtFinePNG/train"

# Define rare class IDs
RARE_CLASSES = [13, 14, 15, 16, 17, 18]  # Car, Truck, Bus, Train, Motorcycle, Bicycle

# Collect image paths with rare classes
rare_class_image_paths = []
MIN_PIXELS = 3000  # or 500

# Track class pixel counts overall (optional)
class_pixel_counts = defaultdict(int)

for city_folder in os.listdir(mask_root):
    city_path = os.path.join(mask_root, city_folder)
    if not os.path.isdir(city_path):
        continue

    for file in os.listdir(city_path):
        if file.endswith(".png"):
            mask_path = os.path.join(city_path, file)
            mask = np.array(Image.open(mask_path))

            unique_classes, counts = np.unique(mask, return_counts=True)
            pixel_dict = dict(zip(unique_classes, counts))

            # Save if any rare class has at least MIN_PIXELS pixels
            has_rare_class = False
            for cls in RARE_CLASSES:
                if pixel_dict.get(cls, 0) >= MIN_PIXELS:
                    has_rare_class = True
                    break

            if has_rare_class:
                rare_class_image_paths.append(mask_path)

            # (Optional) Count total pixels per class
            for cls, cnt in pixel_dict.items():
                class_pixel_counts[cls] += cnt

# Save to file
with open("rare_class_images.txt", "w") as f:
    for path in rare_class_image_paths:
        f.write(path + "\n")

print(f"Saved {len(rare_class_image_paths)} rare-class image paths.")