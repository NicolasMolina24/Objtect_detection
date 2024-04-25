import os
import torch

# Paths to the data, model, images, and output directories
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model')
IMAGES_PATH = os.path.join(os.path.dirname(__file__), 'images')

# Output path
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'output')

# Model parameters
SEED = 24
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY = True if DEVICE == 'cuda' else False

## Pretrained model
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

## Model parameters
LR = 1e-4
N_EPOCHS = 10
BATCH_SIZE = 32

## Loss for the model
L_CLASS = 1.
L_BBOX = 1.

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

TORCH_SEED = seed_everything(SEED)