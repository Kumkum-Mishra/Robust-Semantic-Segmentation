# configs/idd_config.py
from configs.base_config import BaseConfig

class IDDConfig(BaseConfig):
    dataset_root: str = "data/IDD/IDD_Segmentation"   
    left_img_dir: str = f"{dataset_root}/leftImg8bit"
    gt_dir: str = f"{dataset_root}/gtFinePNG"

    model_name: str = "deeplabv3plus"
    backbone: str = "resnet101"
    dataset_percentage: float = 0.01  # Use full data for training
    device: str = "cuda"
    log_dir: str = "logs/idd"
    checkpoint_dir: str = "checkpoints/idd"
    output_dir: str = "outputs/idd"

Config = IDDConfig
