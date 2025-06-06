# configs/iddaw_config.py
from configs.base_config import BaseConfig

class IDDAWConfig(BaseConfig):
    dataset_root: str = "data/IDD/IDDAW/train"  # You test on this in Phase 1
    weather_conditions: list = ["FOG", "RAIN", "LOWLIGHT", "SNOW"]

    model_name: str = "deeplabv3plus"
    backbone: str = "resnet101"
    device: str = "cuda"
    log_dir: str = "logs/iddaw"
    checkpoint_dir: str = "checkpoints/iddaw"
    output_dir: str = "outputs/iddaw"

Config = IDDAWConfig
