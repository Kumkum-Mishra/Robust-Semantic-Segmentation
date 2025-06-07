import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
NUM_CLASSES = 19
CLASS_NAMES = [
    'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
    'Pole', 'Traffic Light', 'Traffic Sign', 'Vegetation',
    'Terrain', 'Sky', 'Person', 'Rider', 'Car', 'Truck',
    'Bus', 'Train', 'Motorcycle', 'Bicycle'
]
# Analyzed on 500 images
def count_pixels_in_folder(label_folder, max_files=500):
    class_counts = defaultdict(int)
    count = 0

    for root, _, files in os.walk(label_folder):
        for file in files:
            if file.endswith(".png"):
                path = os.path.join(root, file)
                label_img = np.array(Image.open(path))
                unique, counts = np.unique(label_img, return_counts=True)
                for cls, cnt in zip(unique, counts):
                    if cls < NUM_CLASSES:
                        class_counts[int(cls)] += cnt
                count += 1
                if max_files and count >= max_files:
                    return class_counts
    return class_counts

def plot_distribution(class_counts, title, save_path):
    counts = [class_counts.get(i, 0) for i in range(NUM_CLASSES)]
    plt.figure(figsize=(12, 6))
    plt.bar(CLASS_NAMES, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel('Pixel Count')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

datasets = {
    "IDD Train": "data/IDD/IDD_Segmentation/gtFinePNG/train",
    "IDD Val": "data/IDD/IDD_Segmentation/gtFinePNG/val",
    "IDD-AW FOG": "data/IDD/IDDAW/train/FOG/gtSegPNG",
    "IDD-AW RAIN": "data/IDD/IDDAW/train/RAIN/gtSegPNG",
    "IDD-AW LOWLIGHT": "data/IDD/IDDAW/train/LOWLIGHT/gtSegPNG",
    "IDD-AW SNOW": "data/IDD/IDDAW/train/SNOW/gtSegPNG"
}

os.makedirs("plots", exist_ok=True)

# === Run analysis and plotting ===
for name, path in datasets.items():
    print(f"Processing {name}...")
    counts = count_pixels_in_folder(path)
    plot_distribution(counts, f"Class Distribution - {name}", f"plots/{name.replace(' ', '_')}.png")
