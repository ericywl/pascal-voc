import numpy as np
import PIL
import random
import matplotlib.pyplot as plt
from pprint import pprint

import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.models as models
import torchvision.transforms as transforms

from vocparse import PascalVOC
from sklearn.metrics import average_precision_score


DEFAULT_DATASET_DIR = "./VOCdevkit/VOC2012/"
SEED = 2019
USE_CUDA = True


class ImageDataset(torch.utils.data.Dataset):
    """Dataset containing image paths, images are opened only when needed"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        label = self.labels[index]
        fname = self.image_paths[index]
        img = PIL.Image.open(self.image_paths[index])
        if self.transform is not None:
            img = self.transform(img)
            label = torch.Tensor(label)
        return img, label, fname

    def __len__(self):
        return len(self.image_paths)


class PascalClassifier:
    """Classifier for Pascal VOC"""
    NUM_CLASSES = 20
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

    def __init__(self, model=None, device=None, weights_path=None, scale=300,
                 crop_sz=280, rotation=30, five_crop=True, means=None, stds=None):
        if not model:
            # Initialize model
            model = models.resnet18(pretrained=True)
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(512, self.NUM_CLASSES)
        self.model = model
        self.device = device if device else torch.device("cpu")
        self.weights = torch.load(weights_path) if weights_path else None
        self.model.to(device)
        # Calculated means and stds from Pascal VOC
        self.means = means if means else [0.4603, 0.4384, 0.4051]
        self.stds = stds if stds else [0.1576, 0.1561, 0.1611]
        # Image parameters
        self._image_params = [scale, crop_sz, rotation]
        self._five_crop = five_crop
        if self._five_crop:
            self._trf = transforms.Compose([
                transforms.Resize(scale),
                transforms.FiveCrop(crop_sz),
                transforms.Lambda(lambda crops: torch.stack(
                    [transforms.Normalize(self.means, self.stds)(
                        transforms.ToTensor()(crop)) for crop in crops])
                )
            ])
        else:
            self._trf = transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(crop_sz),
                transforms.ToTensor(),
                transforms.Normalize(means, stds)
            ])
        # Misc variables
        self._best_loss = -1.0
        self._val_loss_arr = []
        self._train_loss_arr = []

    @property
    def scale(self):
        return self._image_params[0]

    @property
    def crop_sz(self):
        return self._image_params[1]

    @property
    def rotation(self):
        return self._image_params[2]

    def train(self, train_loader, criterion, optimizer):
        """Train the model"""
        self.model.train(mode=True)
        train_loss = 0
        print("TRAIN:")
        for batch_id, (features, labels, _) in enumerate(train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            if self._five_crop:
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

    def val(self, val_loader, criterion, epoch, max_epochs):
        """Perform validation to choose the best weights from epochs"""
        outputs_all = torch.zeros(0, self.NUM_CLASSES).to(self.device)
        labels_all = torch.zeros(0, self.NUM_CLASSES).to(self.device)
        fnames_all = []
        ap_scikit = 0
        val_loss = 0
        self.model.train(mode=False)
        print("VAL:")
        for batch_id, (features, labels, fnames) in enumerate(val_loader):
            with torch.no_grad():
                features = features.to(self.device)
                labels = labels.to(self.device)
                if self._five_crop:
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
                fnames_all.extend(fnames)
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
        if epoch == max_epochs - 1:
            print("Tailacc:")
            # Get tail accuracy
            tailacc = self.get_tailacc(outputs_all, labels_all)
            pprint(tailacc)
            # Save image filenames, outputs, labels and tailacc
            if self._five_crop:
                name = "five_crop"
            else:
                name = "rand_rotate"
            np.save(f"saves/{name}_fnames.npy", np.asarray(fnames_all))
            torch.save(outputs_all, f"saves/{name}_outputs.pth")
            torch.save(labels_all, f"saves/{name}_labels.pth")
            torch.save(tailacc, f"saves/{name}_tailacc.pth")

    def get_tailacc(self, outputs_all, labels_all):
        # Get tail accuracy
        tailacc = dict()
        tmax = outputs_all.max()
        eps = (tmax - 0.5) / self.NUM_CLASSES
        correct_all = outputs_all * labels_all
        for t in range(self.NUM_CLASSES):
            threshold = 0.5 + eps * t
            denom = (outputs_all > threshold).float().sum(dim=0)
            numer = (correct_all > threshold).float().sum(dim=0)
            tailacc[threshold] = numer / denom
        return tailacc

    def run_trainval(self, batch_sz=8, max_epochs=30, learn_rate=0.01,
                     loss_weights=None, dataset_dir=DEFAULT_DATASET_DIR):
        """Run both training and validation for some epochs"""
        # Getting train, val filenames and multi-hot labels
        pv = PascalVOC(dataset_dir)
        train_filenames, train_labels = pv.imgs_to_fnames_labels("train")
        val_filenames, val_labels = pv.imgs_to_fnames_labels("val")
        # Loading dataset
        train_dataset = ImageDataset(train_filenames, train_labels,
                                     transform=self._trf)
        val_dataset = ImageDataset(val_filenames, val_labels,
                                   transform=self._trf)
        train_loader = tdata.DataLoader(train_dataset, batch_size=batch_sz,
                                        shuffle=True)
        val_loader = tdata.DataLoader(val_dataset, batch_size=batch_sz,
                                      shuffle=True)
        # Stochastic gradient descent optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learn_rate)
        # Binary cross entropy with logits, weighted loss function
        if not loss_weights:
            const = min((self.CLASS_OCC_DICT.values()))
            loss_weights = torch.Tensor([
                const * 1.0 / co for co in self.CLASS_OCC_DICT.values()
            ]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(weight=loss_weights)
        # Training and validation
        for epoch in range(max_epochs):
            print(f"Epoch {epoch}")
            print("======================")
            #self.train(train_loader, criterion, optimizer)
            self.val(val_loader, criterion, epoch, max_epochs)
            print()
        # Save best model weights and loss arrays
        if self._five_crop:
            name = "five_crop"
        else:
            name = "rand_rotate"
        torch.save(self.weights, f"weights/{name}_weights.pth")
        np.save(f"saves/{name}_train_loss.npy", self._train_loss_arr)
        np.save(f"saves/{name}_val_loss.npy", self._val_loss_arr)

    def predict(self, image):
        """Predict the labels associated with image in image path"""
        if not self.weights:
            raise ValueError("Weights not found, can't do prediction.")
        self.model.load_state_dict(self.weights)
        self.model.eval()
        image = self._trf(PIL.Image.open(image))
        features = image[None].to(self.device)
        # Perform  prediction
        if self._five_crop:
            bsize, ncrops, chan, height, width = features.size()
            outputs = self.model(
                features.view(-1, chan, height, width))
            outputs = outputs.view(bsize, ncrops, -1).mean(1)
        else:
            outputs = self.model(features)
        output = torch.sigmoid(outputs[0])
        # TODO: Test out prediction method
        # TODO: Integrate prediction with GUI
        return output

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
    pv = PascalVOC(DEFAULT_DATASET_DIR)
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


def random_seeding(seed_value):
    """Seed all random functions for reproducibility"""
    if seed_value > 0:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def test_predict():
    pc = PascalClassifier(weights_path="weights/five_crop_weights.pth")
    pc.predict("sofa-cat.jpg")
    
def top_confidence_list():
    """Rank the top predictions for each class and output as json"""
    f_names=np.load('saves/five_crop_fnames.npy')
    outputs_all=torch.load("saves/five_crop_outputs.pth",map_location=torch.device('cpu'))
    labels_all=torch.load("saves/five_crop_labels.pth",map_location=torch.device('cpu'))
    n,k=outputs_all.shape
    out_reshape=outputs_all.permute(1,0)
    lab_reshape=labels_all.permute(1,0)
    pv = PascalVOC(DEFAULT_DATASET_DIR)
    json_output=dict()
    
    for i in range(k):
        ap_scikit = average_precision_score(
            lab_reshape[i].cpu(), out_reshape[i].cpu())
        predictions = (out_reshape[i]>0.5).float()*out_reshape[i]
        prediction_number = (out_reshape[i]>0.5).sum()
        precision,location = torch.sort(predictions,dim=0,descending=True)
        urls=[f_names[j] for j in location]
        corrects= [lab_reshape[i][j] for j in location]
        json_output[i]={"class_name":pv.list_image_sets()[i],"AP":ap_scikit}
        json_output[i]["images"]=[{"images_url":urls[j],"confidence":precision[j].item(),"correct":corrects[j].item()}for j in range(prediction_number)]
    
    with open('saves/ranks.json', 'w') as outfile:
        json.dump(json_output, outfile, sort_keys = False, indent = 4,ensure_ascii = False)   


def main():
    """Main function"""
    # Seeding if it's available
    random_seeding(SEED)
    # Initialize CUDA device
    device = torch.device("cuda") if USE_CUDA else torch.device("cpu")
    # Run training and validation
    pc = PascalClassifier(
        device=device, weights_path="weights/five_crop_weights.pth")
    pc.run_trainval(max_epochs=1)


if __name__ == "__main__":
    main()
    # test_predict()
    # top_confidence_list()