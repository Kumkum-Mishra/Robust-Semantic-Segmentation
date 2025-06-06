import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from dataloaders.iddaw_loader import IDDAWDataset
from configs.idda_config import Config
from utils.metrics import compute_miou

device = torch.device(Config.device)

# Load Model
model = deeplabv3_resnet101(pretrained=False, num_classes=19)
state_dict = torch.load(Config.model_path)
model.load_state_dict(state_dict, strict=False)  
model.to(device)
model.eval()

# Evaluate on each condition
# Evaluate on each condition
for condition in Config.conditions:
    print(f"\nEvaluating on {condition} data...")
    dataset = IDDAWDataset(
        root=Config.dataset_root,
        condition=condition,
        split="val",
        image_size=Config.image_size,
        subset_percent=0.01  # evaluate only on 1% of data
    )
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, num_workers=Config.num_workers)


    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)["out"]
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    print("Unique predicted values:", torch.unique(torch.cat(all_preds)))
    print("Unique label values:", torch.unique(torch.cat(all_labels)))

    miou = compute_miou(torch.cat(all_preds), torch.cat(all_labels), num_classes=19)
    print(f"mIoU on {condition}: {miou[0]:.4f}")

