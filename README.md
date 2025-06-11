
# Robust Semantic Segmentation under Adverse Weather Conditions

This repository contains all code, configuration, and logs related to my internship project:  
**"Improving Semantic Segmentation under Adverse Weather Conditions using CNN Models on IDD-AW"**.

## 🔍 Project Overview

This project aims to improve semantic segmentation performance under challenging weather conditions (fog, rain, low-light) using CNN-based models like DeepLabV3+. It follows a 6-phase strategy, with completed work up to **Phase 2: Data-Centric Enhancements**.

## Directory Structure

```
├── checkpoints/              # Model weights (best and latest)
├── configs/                  # YAML config files (base, IDD, IDD-AW)
├── data/                     # Contains IDD, IDD-AW datasets (manually placed)
├── dataloaders/             # Custom PyTorch dataset loaders
├── datasets/                # Dataset parsing and label mapping
├── logs/                     # JSON logs of training
├── models/                  # DeepLabV3+ model definition
├── outputs/                 # Evaluation outputs (metrics, confusion matrix)
├── plots/                   # Class distribution and comparison plots
├── utils/                   # Helper modules for metrics and logging
├── scripts/                 # Evaluation and pseudo-labeling scripts
├── requirements.txt         # List of Python dependencies
├── train.py                 # Main training script
├── eval.py                  # Evaluation script
└── README.md                # You're here!
```

## Datasets Used

| Dataset   | Description |
|-----------|-------------|
| **IDD-Segmentation** | Indian Driving Dataset (clear-weather segmentation). Used for training. |
| **IDD-AW**           | IDD under fog, rain, low-light, and snow. Used for evaluation. |

> **Manual Step**: Download datasets from official sources and unzip them into the `/data/` folder as follows:
```
data/
├── IDD_Segmentation/
├── IDDAW/
```

## Environment Setup

1. **Create virtual environment**:
```bash
python -m venv venv
```

2. **Activate virtual environment**:
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. **Install required packages**:
```bash
pip install -r requirements.txt
```

> Includes: `torch`, `albumentations`, `opencv-python`, `numpy`, `matplotlib`, `tqdm`, and others.

## How to Train

**Phase 1 – Baseline CNN Training:**
```bash
python train.py --config configs/idd_config.py
```

- Trains DeepLabV3+ on clear-weather images from IDD
- Stores results and logs in `checkpoints/` and `logs/`

**Phase 2 – Data-Centric Rare Class Training:**
```bash
python train.py --config configs/idd_config.py --rare-class-only
```

- Uses filtered dataset with rare class images only  
- Boosts performance on underrepresented classes

## How to Evaluate on Adverse Weather

```bash
python eval.py --config configs/iddaw_config.py
```

- Evaluates on IDD-AW test subsets: FOG, RAIN, LOWLIGHT, and SNOW
- Logs mIoU and class-wise IoU in `outputs/` and `logs/`

## TensorBoard Logs

To visualize training metrics:
```bash
tensorboard --logdir=runs/
```

## Manual vs Automatic Tasks

| Task | Manual | Automatic |
|------|--------|-----------|
| Dataset download & unzip | ✅ | ❌ |
| JSON to PNG label conversion | ✅ | ✅ (Scripts available) |
| Training | ❌ | ✅ |
| Evaluation & logging | ❌ | ✅ |
| Visualization plots | ✅ (Run script) | ✅ (Plots saved automatically) |

## 🔗 GitHub Repository

You can find all code, scripts, and logs at:  
[🔗 GitHub – Robust Semantic Segmentation](https://github.com/Kumkum-Mishra/Robust-Semantic-Segmentation)

## Contact

For any queries related to the repository or the internship project, feel free to reach out via GitHub or email.
