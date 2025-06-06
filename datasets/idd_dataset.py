# datasets/idd_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class IDDAWDataset(Dataset):
    def __init__(self, root, condition, split="val", image_size=(512, 1024), subset_percent=0.01):
        self.rgb_dir = os.path.join(root, split, condition, "rgb")
        self.label_dir = os.path.join(root, split, condition, "gtSegPNG")
        self.image_size = image_size

        self.image_paths = []
        self.label_paths = []

        for folder in os.listdir(self.rgb_dir):
            folder_path = os.path.join(self.rgb_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            label_folder = os.path.join(self.label_dir, folder)
            if not os.path.exists(label_folder):
                continue

            for file in os.listdir(folder_path):
                if file.endswith("_rgb.png"):
                    base_id = file.replace("_rgb.png", "")
                    rgb_path = os.path.join(folder_path, file)
                    mask_filename = base_id + "_mask.png"
                    mask_path = os.path.join(label_folder, mask_filename)

                    if os.path.exists(mask_path):
                        self.image_paths.append(rgb_path)
                        self.label_paths.append(mask_path)

        # Reduce to subset_percent
        total = len(self.image_paths)
        subset_size = max(1, int(total * subset_percent))
        self.image_paths = self.image_paths[:subset_size]
        self.label_paths = self.label_paths[:subset_size]



        assert len(self.image_paths) == len(self.label_paths), "Image-label mismatch!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
       image = Image.open(self.image_paths[idx]).convert("RGB")
       label = Image.open(self.label_paths[idx])

       # Ensure both are resized
       image = image.resize(self.image_size, Image.BILINEAR)
       label = label.resize(self.image_size, Image.NEAREST)

       if self.transform:
           transformed = self.transform(image=image, mask=label)
           image = transformed["image"]
           label = transformed["mask"]
       else:
           image = T.ToTensor()(image)
           label = T.PILToTensor()(label).squeeze(0).long()  # Shape: [H, W]

       return image, label
