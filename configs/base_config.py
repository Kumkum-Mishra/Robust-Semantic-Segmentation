# configs/base_config.py
import os
from dataclasses import dataclass, field

@dataclass
class BaseConfig:
    # Core paths
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"

    # Device
    device: str = "cpu"  

    # Training hyperparameters
    batch_size: int = 2
    num_epochs: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    patience: int = 5

    # Image and data properties
    image_size: tuple = (512, 1024)
    num_workers: int = 0
    ignore_index: int = 255
    num_classes: int = 19

    # Dataset usage controls
    dataset_percentage: float = 0.009  # Use 1.0 for full training
    augment_prob: float = 0.8        # For Phase 2 augmentations

    # Model selection (used in Phase 1 to 5)
    model_name: str = "deeplabv3plus"  # deeplabv3plus, hrnet, swiftnet, segformer, swin
    backbone: str = "resnet101"        # resnet101, mobilenetv2, resnext50, etc.

    # Flags for toggling advanced features (used later in Phase 2-5)
    use_tta: bool = False              # Enable Test-Time Augmentation
    use_fusion: bool = False           # Enable fusion modules (Phase 5)
    use_edge_head: bool = False        # Auxiliary edge prediction

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

# Export shortcut for import
Config = BaseConfig
