import os
import glob
import numpy as np
import PIL
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms

from vocparse import PascalVOC


DATASET_DIR = "./VOCdevkit/VOC2012/"


class ImageDataset(torch.utils.data.Dataset):
    """Dataset containing image paths, images are opened only when needed"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        img = PIL.Image.open(self.image_paths[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_paths)

# ======= From my previous homeworks ======= #
# ====== Will need changes to fit VOC ====== #


def train(model, device, train_loader, criterion, optimizer, epoch, loss_arr):
    """Train the model"""
    model.train(mode=True)
    train_loss = 0
    print(f"Train epoch: {epoch}")
    for batch_id, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        # Calculate loss
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        # Compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        if batch_id % 100 == 0:
            print("[{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                batch_id * len(features), len(train_loader.dataset),
                100. * batch_id / len(train_loader), loss.item()))
    train_loss /= len(train_loader)
    print(f"Average training loss: {train_loss}")
    loss_arr.append(train_loss)


def val(model, device, val_loader, criterion, best_loss, best_weights, loss_arr):
    """Perform validation to choose the best weights from epochs"""
    model.train(mode=False)
    val_loss = 0
    correct = 0
    for features, labels in val_loader:
        with torch.no_grad():
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            val_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1, keepdim=True)
            correct += preds.eq(labels.view_as(preds)).sum().item()
    val_loss /= len(val_loader)
    print(f"Average validation loss: {val_loss}")
    print(f"Validation accuracy: {correct / len(val_loader.dataset)}\n")
    loss_arr.append(val_loss)
    # Replace best_weights with current weights if validation loss is better
    if (best_loss < 0) or (val_loss < best_loss):
        best_loss = val_loss
        best_weights = model.state_dict()
    return best_weights

# ========================================== #


def main():
    # Parameters
    batch_sz = 8
    scale = 256
    crop_sz = 224
    ## Need to recalculate means for the dataset
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    # Loading dataset
    pv = PascalVOC(DATASET_DIR)
    train_filenames, train_labels = pv.imgs_to_fnames_labels("train")
    val_filenames, val_labels = pv.imgs_to_fnames_labels("val")
    trf = transforms.Compose([
        transforms.Resize(scale),
        transforms.FiveCrop(crop_sz),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
    train_dataset = ImageDataset(train_filenames, train_labels,
                                 transform=trf)
    val_dataset = ImageDataset(val_filenames, val_labels, transform=trf)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_sz)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_sz)


if __name__ == "__main__":
    main()
