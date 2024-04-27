import os
import torch
from pathlib import Path

# Paths to the data, model directories
DATASET_PATH = Path('..\data')
DATASET_PATH.mkdir(exist_ok=True)
MODEL_PATH = Path('..\models')
MODEL_PATH.mkdir(exist_ok=True)
# Path to the data
CSV_PATH = DATASET_PATH.joinpath('data.csv')

# Output path
OUTPUT_PATH = Path('..\output')
# Create the output directory
OUTPUT_PATH.mkdir(exist_ok=True)
# Accuracy and loss plot name
PLOT_NAME = 'loss_accuracy.png'
MODEL_NAME = 'best_model_{epoch}.pth'
# Model parameters
SEED = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = True if DEVICE == 'cuda' else False

## Pretrained model
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

## Model parameters
LR = 1e-4
EPOCHS = 10
BATCH_SIZE = 32

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

TORCH_SEED = seed_everything(SEED)
SET_SEED = False