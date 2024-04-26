# USAGE
# python train.py
# import the necessary packages
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss, lr_scheduler
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch
import os
import model
import config
import dataloader
import utils


def train_step(model:torch.nn.Model, data_loader, loss_fn, optimizer, accuracy_fc, device):
    # loop through the training set
    model.train()
    train_loss, train_accuracy = 0.0, 0.0
    for i, (image, label, bbox) in enumerate(data_loader):
        # 0. data to device
        image, label, bbox = image.to(device), label.to(device), bbox.to(device)
        # 1. Forward pass
        prediction = model(image)
        # 2. Compute loss and accuracy
        total_loss = (loss_fn['bbox_loss'](prediction[0], bbox) + 
                      loss_fn['class_loss'](prediction[1], label))
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
            test_loss += (loss_fn['bbox_loss'](prediction[0], bbox) +
                          loss_fn['class_loss'](prediction[1], label))
            test_acc += accuracy_fn(y_true=target, y_pred=class_prediction)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
            
    return test_loss, test_acc

def train_model(seed=config.SET_SEED, scheduler=False) -> None:

    if seed:
        # set the manual seed for reproducibility
        config.TORCH_SEED

    # create csv file with the data
    csv_data = dataloader.create_data_csv(config.DATASET_PATH)

    # load the dataLOader
    trainDataset = dataloader.FewShotDataset(csv_data, dataloader.transforms()['train'])
    testDataset = dataloader.FewShotDataset(csv_data, dataloader.transforms()['test'])
    # get the number of classes in the train dataset
    num_classes = trainDataset.get_classes()

    # create data loaders
    trainLoader = DataLoader(trainDataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
    testLoader = DataLoader(testDataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)
        
    # load the pretrain model 
    baseModel = resnet50(pretrained=True)
    # freeze the model
    for param in baseModel.parameters():
        param.requires_grad = False
    # create the object save_best_model
    save_best_model = utils.SaveBestModel(config.MODEL_PATH)
    # create the model
    objectDetector = model.FewShotModel(baseModel, num_classes=num_classes)
    # send the model to the device
    objectDetector.to(config.DEVICE)

    # define the loss for bbox and class
    loss_fn = {
        'bbox_loss': MSELoss(),
        'class_loss': CrossEntropyLoss()
    }

    # define the optimizer
    optimizer = Adam(objectDetector.parameters(), lr=config.LR)
    if scheduler:
        lr_sr = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in tqdm(range(config.EPOCHS)):
        # train step
        train_loss = train_step(objectDetector, trainLoader, loss_fn, optimizer, f1_score, config.DEVICE)
        # test step
        val_loss, accuracy = test_step(objectDetector, trainLoader, loss_fn, f1_score, config.DEVICE)

        history['train_loss'].append(train_loss[0])
        history['train_accuracy'].append(train_loss[1])
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)

        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}')

        # adjust the learning rate
        if scheduler:
            lr_sr.step()
            print(f'Current learning rate: {lr_sr.get_last_lr()}')

        # save the best model on validation loss
        save_best_model(val_loss, epoch, objectDetector, optimizer)

        # save the plots for loss and accuracy
        utils.save_plot(history, config.PLOT_NAME)
