import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class RareClassIDDSubset(Dataset):
    def __init__(self, image_paths_file, image_root, mask_root, transform=None, max_samples=None):
        with open(image_paths_file, 'r') as f:
            self.mask_paths = [line.strip() for line in f.readlines()]
        if max_samples is not None:
            self.mask_paths = self.mask_paths[:max_samples]

        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):  # <-- FIXED: now inside the class!
        mask_path = self.mask_paths[idx]
        img_base = mask_path.replace('gtFinePNG', 'leftImg8bit') \
                            .replace('gtFine_polygons.png', 'leftImg8bit')
        img_path_png = img_base + '.png'
        img_path_jpg = img_base + '.jpg'

        if os.path.exists(img_path_png):
            img_path = img_path_png
        elif os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        else:
            raise FileNotFoundError(f"Image not found: {img_path_png} or {img_path_jpg}")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = image.resize((512, 1024), Image.BILINEAR)
        mask = mask.resize((512, 1024), Image.NEAREST)

        image = T.ToTensor()(image)
        mask = T.PILToTensor()(mask).long().squeeze(0)

        return image, mask

def get_rare_class_dataloaders(config,max_samples=40):
    image_paths_file = "rare_class_images.txt"
    image_root = config.left_img_dir
    mask_root = config.gt_dir

    train_dataset = RareClassIDDSubset(
        image_paths_file=image_paths_file,
        image_root=image_root,
        mask_root=mask_root,
        max_samples=max_samples
    )
    val_dataset = RareClassIDDSubset(
        image_paths_file=image_paths_file,
        image_root=image_root,
        mask_root=mask_root,
        max_samples=max_samples
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    return train_loader, val_loader