from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configs.idd_config import IDDConfig
writer = SummaryWriter(log_dir=os.path.join("runs", IDDConfig.model_name))

from models.deeplabv3plus import get_deeplabv3plus
from utils.metrics import compute_miou
from utils.logger import save_model, log_to_console

# IMPORT your new dataloader for rare-class images
from dataloaders.rare_class_subset_loader import get_rare_class_dataloaders

def train():
    # Load config and device
    config = IDDConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else config.device)

    # Load data using rare-class loader
    train_loader, val_loader = get_rare_class_dataloaders(config, max_samples=40)

    # Load model
    model = get_deeplabv3plus(config.num_classes, backbone=config.backbone)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.patience)

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Training"):
            images, masks = images.to(device), masks.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, dict):  # Fix for torchvision models
                outputs = outputs["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} - Validation"):
                images, masks = images.to(device), masks.to(device).long()
                outputs = model(images)
                if isinstance(outputs, dict):  # Fix for torchvision models
                    outputs = outputs["out"]
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        log_to_console(epoch, train_loss, val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config.checkpoint_dir, config.model_name + "_rareclass")

if __name__ == "__main__":
    train()
