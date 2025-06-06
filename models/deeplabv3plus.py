# models/deeplabv3plus.py
import torch
import torchvision.models.segmentation as models

def get_deeplabv3plus(num_classes: int, backbone: str = "resnet101"):
    if backbone == "resnet101":
        model = models.deeplabv3_resnet101(pretrained=True)
    elif backbone == "resnet50":
        model = models.deeplabv3_resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # Replace the classifier head
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    return model