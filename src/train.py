# USAGE
# python train.py
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
import cv2
import os
import model
import config
import dataloader
import ultils

# initialize the list of data (images), class labels, target bounding
# box coordinates, and image paths
print("[INFO] loading dataset...")
data = []
labels = []
bboxes = []
imagePaths = []


def train_step(model:torch.nn.Model, data_loader, loss_fn, optimizer, accuracy_fc, device):
    # loop through the training set
    model.train()
    train_loss, train_accuracy = 0.0, 0.0
    for i, (image, label, bbox) in enumerate(data_loader):
        # 0. data to device
        image, label, bbox = image.to(device), label.to(device), bbox.to(device)
        # 1. Forward pass
        prediction = model(data)
        # 2. Compute loss and accuracy
        total_loss = loss_fn['bbox_loss'](prediction[0], bbox) + loss_fn['class_loss'](prediction[1], label)
        train_loss += total_loss.item()

        train_accuracy += accuracy_fc(prediction[1].argmax(1), label)
        # 3. Optimizer zero_grad
        optimizer.zero_grad()
        # 4. Backprop
        total_loss.backward()
        # 5. Optimizer step
        optimizer.step()

    train_loss /= len(data_loader)
    train_accuracy /= len(data_loader)
            
    return train_loss, train_accuracy

def test_step(model, data_loader, loss_fn, accuracy_fn, device):
    # Set model evaluation mode
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for i, (image, label, bbox) in enumerate(data_loader):
            # 0. data to device
            image, label, bbox = image.to(device), label.to(device), bbox.to(device)
            data, target = data.to(device), target.to(device)
            # 1. Forward pass
            prediction = model(data)
            # from logits to class
            class_prediction = prediction.argmax(dim=1)
            # 2. Compute loss and accuracy
            test_loss += loss_fn['bbox_loss'](prediction[0], bbox) + loss_fn['class_loss'](prediction[1], label)
            test_acc += accuracy_fn(y_true=target, y_pred=class_prediction)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
            
    return test_loss, test_acc

def train_model():

    # set the manual seed for reproducibility
    config.TORCH_SEED

    # load the dataLOader
    trainDataset = dataloader.FewShotDataset('test.csv', dataloader.transforms()['train'])

    # create data loaders
    trainLoader = DataLoader(trainDataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
        

    # load the pretrain model 
    baseModel = resnet50(pretrained=True)
    # freeze the model
    for param in baseModel.parameters():
        param.requires_grad = False
    
    save_best_model = model.SaveBestModel('D:/best_model')

    objectDetector = model.FewShotModel(baseModel, num_classes=4) # fix this
    # send the model to the device
    objectDetector.to(config.DEVICE)

    # define the loss for bbox and class
    loss_fn = {
        'bbox_loss': MSELoss(),
        'class_loss': CrossEntropyLoss()
    }

    # define the optimizer
    optimizer = Adam(objectDetector.parameters(), lr=config.LR)

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in tqdm(range(config.EPOCHS)):
        # train step
        train_loss = train_step(model, trainLoader, loss_fn, optimizer, f1_score, config.DEVICE)
        # test step
        val_loss, accuracy = test_step(model, trainLoader, loss_fn, f1_score, config.DEVICE)

        history['train_loss'].append(train_loss[0])
        history['train_accuracy'].append(train_loss[1])
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)

        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}')

        # save the best model on validation loss
        save_best_model(val_loss, epoch, model, optimizer)

        # save the plots for loss and accuracy
        ultils.save_plot(history, 'loss')
    return model