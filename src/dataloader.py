from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from torchvision import transforms
import config


def create_csv_data(path_data: str) -> str:
    """
    Create a CSV file with columns image, label, bbox and use
    image is path to the image, label is the class of the image,
    bbox are the coordinates of the bounding box of the image and 
    use is the type of data (train, test, val)
    
    Args:
    path_data (Path): path to the data directory

    Returns:
    str: path to the CSV file
    """
    data = []
    uses = ['test', 'train', 'val']
    for use in uses:
        path_data_use = path_data.joinpath(use)
        for image_file in path_data_use.joinpath('images').glob('*'):
            file_name = image_file.stem
            up_folder = path_data_use.joinpath('labels')
            label_file = up_folder.joinpath(file_name + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    label = f.read()
                    label = label.split(' ')
                    label, bbox = label[0], label[1:]
                data.append({'image': image_file, 'label': label, 'bbox': bbox, 'use': use})
            else:
                continue
    data = pd.DataFrame(data)
    data.to_csv(config.CSV_PATH, index=False)
    return config.CSV_PATH


def transformations() -> dict:
    """
    Create a dictionary with the transformations to apply to the images

    Returns:
    dict: dictionary with the transformations    
    """
    set_transformations = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
    }
    return set_transformations

class FewShotDataset(Dataset):
    """
    Dataset class for Few-Shot Learning
    
    Args:
    data_path (str): path to the CSV file with the data
    transform (torchvision.transforms): transform to apply to the images
    
    Returns:
    torch.utils.data.Dataset: Dataset object
    """
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image']
        label = sample['label']
        bbox = sample['bbox']

        # load the image
        image = Image.open(image)

        if self.transform:
            image = self.transform(image)
        return image, label, bbox

    def get_classes(self):
        return self.data['label'].nunique()