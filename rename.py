import os

def rename_leftImg8bit_to_gtFine(gt_dir):
    for split in ["train", "val"]:
        split_dir = os.path.join(gt_dir, split)
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.endswith("_leftImg8bit.png"):
                    base = file.replace("_leftImg8bit.png", "")
                    new_name = base + "_gtFine_polygons.png"
                    old_path = os.path.join(root, file)
                    new_path = os.path.join(root, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} → {new_path}")

# Example usage
gt_dir = "data/IDD/IDD_Segmentation/gtFinePNG"
rename_leftImg8bit_to_gtFine(gt_dir)