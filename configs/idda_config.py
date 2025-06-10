from configs.base_config import BaseConfig

class IDDAConfig(BaseConfig):
    dataset_root: str = "data/IDD/IDDAW"
    #model_path: str = "checkpoints/deeplabv3plus.pth"  # Path to our saved model  ( phase 1)
    model_path: str = "checkpoints/deeplabv3plus_rareclass.pth"   # phase 2

    device: str = "cpu"
    log_dir: str = "logs/iddaw"
    output_dir: str = "outputs/iddaw"

    conditions: list = ["FOG", "RAIN", "LOWLIGHT", "SNOW"]

Config = IDDAConfig
