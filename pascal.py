import numpy as np
import PIL
import matplotlib.pyplot as plt
from pprint import pprint

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.models as models
import torchvision.transforms as transforms
import torchnet.meter as meter

from vocparse import PascalVOC
from sklearn.metrics import average_precision_score


DATASET_DIR = "./VOCdevkit/VOC2012/"
NUM_CLASSES = 20
FIVE_CROP = True
SEED = 2019
CLASS_OCC_DICT = {
    "aeroplane": 670,
    "bicycle": 552,
    "bird": 765,
    "boat": 508,
    "bottle": 706,
    "bus": 421,
    "car": 1161,
    "cat": 1080,
    "chair": 1119,
    "cow": 303,
    "diningtable": 538,
    "dog": 1286,
    "horse": 482,
    "motorbike": 526,
    "person": 4087,
    "pottedplant": 527,
    "sheep": 325,
    "sofa": 507,
    "train": 544,
    "tvmonitor": 575
}


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
            label = torch.Tensor(label)
        return img, label

    def __len__(self):
        return len(self.image_paths)


class PascalClassifier:
    def __init__(self, model, device=None, pretrained_weights=None):
        self.model = model
        self.device = device if device else torch.device("cpu")
        self.weights = pretrained_weights
        self.model.to(device)
        # Hidden variables
        self._best_loss = -1.0
        self._val_loss_arr = []
        self._train_loss_arr = []

    def train(self, train_loader, criterion, optimizer):
        """Train the model"""
        self.model.train(mode=True)
        train_loss = 0
        print("TRAIN:")
        for batch_id, (features, labels) in enumerate(train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            if FIVE_CROP:
                bsize, ncrops, chan, height, width = features.size()
                outputs = self.model(features.view(-1, chan, height, width))
                outputs = outputs.view(bsize, ncrops, -1).mean(1)
            else:
                outputs = self.model(features)
            # Calculate loss
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            # Compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            if batch_id % 200 == 0 or batch_id == len(train_loader) - 1:
                if batch_id == len(train_loader) - 1:
                    num = len(train_loader.dataset)
                else:
                    num = batch_id * len(features)
                print("[{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    num, len(train_loader.dataset),
                    100. * batch_id / len(train_loader), loss.item()))
        train_loss /= len(train_loader)
        print(f"Average training loss: {train_loss}")
        self._train_loss_arr.append(train_loss)

    def val(self, val_loader, criterion):
        """Perform validation to choose the best weights from epochs"""
        # AP = torch.zeros(NUM_CLASSES)
        outputs_all = torch.zeros(0, NUM_CLASSES).to(self.device)
        labels_all = torch.zeros(0, NUM_CLASSES).to(self.device)
        ap_scikit = 0
        self.model.train(mode=False)
        val_loss = 0
        print("VAL:")
        for batch_id, (features, labels) in enumerate(val_loader):
            with torch.no_grad():
                features = features.to(self.device)
                labels = labels.to(self.device)
                if FIVE_CROP:
                    bsize, ncrops, chan, height, width = features.size()
                    outputs = self.model(
                        features.view(-1, chan, height, width))
                    outputs = outputs.view(bsize, ncrops, -1).mean(1)
                else:
                    outputs = self.model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                # Concatenate output and labels to large tensor
                new_outputs = torch.sigmoid(outputs)
                outputs_all = torch.cat((outputs_all, new_outputs))
                labels_all = torch.cat((labels_all, labels))
                if batch_id % 200 == 0 or batch_id == len(val_loader) - 1:
                    if batch_id == len(val_loader) - 1:
                        num = len(val_loader.dataset)
                    else:
                        num = batch_id * len(features)
                    print("[{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        num, len(val_loader.dataset),
                        100. * batch_id / len(val_loader), loss.item()))
        val_loss /= len(val_loader)
        print(f"Average validation loss: {val_loss}")
        self._val_loss_arr.append(val_loss)
        # Replace best_weights with current weights if validation loss is better
        if (self._best_loss < 0) or (val_loss < self._best_loss):
            self._best_loss = val_loss
            self.weights = self.model.state_dict()
        # Print AP and accuracy
        ap_scikit = average_precision_score(
            labels_all.cpu(), outputs_all.cpu())
        print(f"AP: {ap_scikit}")
        acc = ((outputs_all > 0.5).float() *
               labels_all).sum() / labels_all.sum()
        print(f"Accuracy: {acc}")
        # Get tail accuracy
        labels_all = outputs_all * labels_all
        tailacc = dict()
        tmax = outputs_all.max()
        print(tmax)
        eps = (tmax - 0.5) / NUM_CLASSES
        for t in range(NUM_CLASSES):
            threshold = 0.5 + eps * t
            confidence = (outputs_all > threshold).sum(dim=0)
            accuracy = (labels_all > threshold).sum(dim=0)
            tailacc[threshold] = accuracy / confidence
        pprint(tailacc)

    def run_trainval(self, optimizer, criterion, scale=380, crop_sz=360,
                     batch_sz=8, max_epochs=30):
        """Run both training and validation for some epochs"""
        # Calculated means and stds from Pascal VOC
        means = [0.4603, 0.4384, 0.4051]
        stds = [0.1576, 0.1561, 0.1611]
        # Getting train, val filenames and multi-hot labels
        pv = PascalVOC(DATASET_DIR)
        train_filenames, train_labels = pv.imgs_to_fnames_labels("train")
        val_filenames, val_labels = pv.imgs_to_fnames_labels("val")
        # Loading dataset
        trf = transforms.Compose([
            transforms.Resize(scale),
            transforms.FiveCrop(crop_sz),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.Normalize(means, stds)(
                    transforms.ToTensor()(crop)) for crop in crops])
            )
        ])
        train_dataset = ImageDataset(train_filenames, train_labels,
                                     transform=trf)
        val_dataset = ImageDataset(val_filenames, val_labels, transform=trf)
        train_loader = tdata.DataLoader(train_dataset, batch_size=batch_sz,
                                        shuffle=True)
        val_loader = tdata.DataLoader(val_dataset, batch_size=batch_sz,
                                      shuffle=True)
        # Training and validation
        for epoch in range(max_epochs):
            print(f"Epoch {epoch}")
            print("======================")
            self.train(train_loader, criterion, optimizer)
            self.val(val_loader, criterion)

    def predict(self):
        # TODO: Integrate prediction with GUI
        pass


# ====== Calculating mean and std for dataset ===== #
# Not used in runtime

def get_means_stds(dataset):
    """Get mean and standard deviation, given dataset"""
    batch_size = 512
    loader = tdata.DataLoader(dataset, batch_size=batch_size,
                              num_workers=1, shuffle=False)
    means = torch.Tensor([0., 0., 0.]).cuda()
    stds = torch.Tensor([0., 0., 0.]).cuda()
    num_samples = float(len(dataset))
    for data, _ in loader:
        data = data.cuda()
        batch_size = data.size(0)
        data = data.view(batch_size, data.size(1), -1)
        # Add up mean and std
        means += data.mean(2).sum(0)
        stds += data.std(2).sum(0)
    # Divide them by number of samples
    means /= num_samples
    stds /= num_samples
    return means, stds


def pascal_means_stds(scale, crop_sz):
    """Calculate mean and standard deviation for Pascal VOC"""
    pv = PascalVOC(DATASET_DIR)
    fnames, labels = pv.imgs_to_fnames_labels("trainval")
    trf = transforms.Compose([
        transforms.Resize(scale),
        transforms.FiveCrop(crop_sz),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops]).mean(0)
        )
    ])
    dataset = ImageDataset(fnames, labels, transform=trf)
    means, stds = get_means_stds(dataset)
    print(means, stds)

# ================================================= #


def main():
    # Seeding if it's available
    if SEED > 0:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
    # Initialize model
    model = models.resnet18(pretrained=True)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(512, NUM_CLASSES)
    # Initialize CUDA device
    device = torch.device("cuda")
    # Stochastic gradient descent optimizer
    learn_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    # Binary cross entropy with logits, weighted loss function
    const = min((CLASS_OCC_DICT.values()))
    weights = torch.Tensor([
        const * 1.0 / co for co in CLASS_OCC_DICT.values()]).to(device)
    criterion = nn.BCEWithLogitsLoss(weight=weights)
    # Run training and validation
    pc = PascalClassifier(model, device)
    pc.run_trainval(optimizer, criterion)


if __name__ == "__main__":
    main()
