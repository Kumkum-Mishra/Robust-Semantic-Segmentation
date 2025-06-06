import os

root_dir = "data/IDD/IDD_Segmentation/gtFinePNG"

for split in ["train", "val"]:
    split_dir = os.path.join(root_dir, split)
    for folder in os.listdir(split_dir):
        folder_path = os.path.join(split_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith("_labelIds.png"):
                os.remove(os.path.join(folder_path, file))
