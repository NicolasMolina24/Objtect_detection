from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import config


def create_csv_data(images_path, labels_path, exts=['jpg', 'jpeg', 'png']) -> str:
    """
    Create a CSV file with the image paths and labels

    Args:
    images_path (str): path to the images directory
    labels_path (str): path to the labels directory
    exts (list): list of image file extensions

    Returns:
    str: path to the CSV file    
    """
    # get labels names files from path
    labels = {label.name:label  for label in list(labels_path.glob('*'))}

    # get list of images in formats jpg, jpeg, png
    images = []
    for ext in exts:
        images.extend(list(images_path.glob(f'*.{ext}')))

    # create a list of dictionaries with the image path and label
    data = []
    for image in images:
        label = image.parent.name
        # read the label file and get the class and bbox
        with open(labels[label], 'r') as f:
            Label_bbox = f.read()
            Label_bbox = Label_bbox.split(' ')
            label, bbox = Label_bbox[0], Label_bbox[1:]
            
        data.append({'image': str(image), 'label': label, 'bbox': bbox})
    # return df with columns image, label, bbox
    data = pd.DataFrame(data)
    data.to_csv(images_path / 'data.csv', index=False)
    return images_path / 'data.csv'


def transforms() -> dict:
    """
    Create a dictionary with the transformations to apply to the images

    Returns:
    dict: dictionary with the transformations    
    """
    transformations = {
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
    return transformations

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