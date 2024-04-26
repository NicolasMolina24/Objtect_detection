from matplotlib import pyplot as plt
from pathlib import Path
from typing import Dict
import config 
import torch

def save_plot(history: Dict, save_path: Path) -> None:
    """
    Save the plot of loss and accuracy

    Args:
    history: dict, the history of loss and accuracy
    save_path: str, the path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train_loss', color='r')
    plt.plot(history['valid_loss'], label='valid_loss', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train_acc', color='r')
    plt.plot(history['valid_acc'], label='valid_acc', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_path.joinpath(config.PLOT_NAME))
    plt.close()

class SaveBestModel():
    """
    Save the best model
    """
    def __init__(self, save_path, best_loss=float('inf')):
        self.best_loss = best_loss
        self.save_path = Path(save_path)

    def __call__(self, valid_loss, epoch, model, optimizer):
        if valid_loss > self.best_loss:
            self.best_loss = valid_loss
            data = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': valid_loss
            }
            torch.save(data, self.save_path.joinpaht(config.MODEL_NAME.format(epoch=epoch)))