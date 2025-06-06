# dataloaders/get_test_dataloaders.py
import os
from torch.utils.data import DataLoader
from datasets.idd_dataset import IDDSegmentationDataset

def get_iddaw_test_loaders(config):
    from torchvision import transforms
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(config.image_size[0], config.image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    subsets = ['FOG', 'RAIN', 'LOWLIGHT', 'SNOW']
    loaders = {}
    for subset in subsets:
        rgb_dir = os.path.join(config.iddaw_root, 'val', subset, 'rgb')
        label_dir = os.path.join(config.iddaw_root, 'val', subset, 'gtSegPNG')
        dataset = IDDSegmentationDataset(rgb_dir, label_dir, image_size=config.image_size, transform=transform)
        loaders[subset] = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    return loaders