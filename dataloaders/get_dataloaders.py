# dataloaders/get_dataloaders.py
import os
from torch.utils.data import DataLoader, Subset
import random
from datasets.idd_dataset import IDDSegmentationDataset
from torchvision import transforms as T

def get_transforms(image_size, augment_prob=0.8):
    basic_transforms = {
        "image": T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]),
        "mask": T.Compose([
            T.Resize(image_size),
            T.PILToTensor()
        ])
    }
    return basic_transforms

def get_dataloaders(config):
    # Clear-weather (IDD-Segmentation)
    train_image_dir = os.path.join(config.left_img_dir, "train")
    val_image_dir = os.path.join(config.left_img_dir, "val")

    train_label_dir = os.path.join(config.gt_dir, "train")
    val_label_dir = os.path.join(config.gt_dir, "val")

    # Basic Resize & Normalize transforms
    transforms_dict = get_transforms(config.image_size)

    # Init datasets
    train_dataset = IDDSegmentationDataset(
        train_image_dir, train_label_dir,
        image_size=config.image_size,
        transform=None  # We can plug albumentations later
    )
    val_dataset = IDDSegmentationDataset(
        val_image_dir, val_label_dir,
        image_size=config.image_size,
        transform=None
    )

    # Reduce dataset size for faster testing if needed
    if config.dataset_percentage < 1.0:
        train_size = int(config.dataset_percentage * len(train_dataset))
        val_size = int(config.dataset_percentage * len(val_dataset))
        train_indices = random.sample(range(len(train_dataset)), train_size)
        val_indices = random.sample(range(len(val_dataset)), val_size)
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader
