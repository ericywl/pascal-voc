import matplotlib.pyplot as plt
import torch
import json
import numpy as np
from sklearn.metrics import average_precision_score

from vocparse import PascalVOC

# ====== Generate json file for Flask web server ===== #
# Not used in runtime


def top_confidence_list():
    """Rank the top predictions for each class and output as json"""
    f_names = np.load('saves/five_crop_fnames.npy')
    outputs_all = torch.load("saves/five_crop_outputs.pth",
                             map_location=torch.device('cpu'))
    labels_all = torch.load("saves/five_crop_labels.pth",
                            map_location=torch.device('cpu'))
    _, k = outputs_all.shape
    out_reshape = outputs_all.permute(1, 0)
    lab_reshape = labels_all.permute(1, 0)
    pv = PascalVOC("./VOCdevkit/VOC2012")
    json_output = dict()

    for i in range(k):
        ap_scikit = average_precision_score(
            lab_reshape[i].cpu(), out_reshape[i].cpu(), average="weighted")
        predictions = (out_reshape[i] > 0.5).float() * out_reshape[i]
        prediction_number = (out_reshape[i] > 0.5).sum()
        precision, location = torch.sort(predictions, dim=0, descending=True)
        urls = [f_names[j] for j in location]
        corrects = [lab_reshape[i][j] for j in location]
        json_output[i] = {
            "class_name": pv.list_image_sets()[i],
            "AP": ap_scikit
        }
        json_output[i]["images"] = [
            {
                "images_url": urls[j],
                "confidence":precision[j].item(),
                "correct":corrects[j].item()
            } for j in range(prediction_number)
        ]

    with open('saves/ranks.json', 'w') as outfile:
        json.dump(json_output, outfile, sort_keys=False,
                  indent=4, ensure_ascii=False)

# ==================================================== #

# ====== Plot all loss and accuracy graphs for report ===== #
# Not used in runtime


def plot(name):
    tail_acc = torch.load(
        f"saves/{name}_tailacc.pth", map_location=torch.device('cpu'))
    train_loss = np.load(f"saves/{name}_train_loss.npy")
    val_loss = np.load(f"saves/{name}_val_loss.npy")
    accuracy = np.load(f"saves/{name}_acc.npy")
    ap = np.load(f"saves/{name}_ap.npy")

    x = np.arange(len(val_loss))
    # Val loss
    plt.plot(x, val_loss)
    plt.xlabel("epoch")
    plt.title(f"{name}_validation_loss")
    plt.savefig(f"plots/{name}_val_loss.jpg")
    plt.close()
    # Train loss
    plt.plot(x, train_loss)
    plt.xlabel("epoch")
    plt.title(f"{name}_training_loss")
    plt.savefig(f"plots/{name}_train_loss.jpg")
    plt.close()
    # Accuracy
    plt.plot(x, accuracy)
    plt.xlabel("epoch")
    plt.title(f"{name}_accuracy")
    plt.savefig(f"plots/{name}_acc.jpg")
    plt.close()
    # Average precision
    plt.plot(x, ap)
    plt.xlabel("epoch")
    plt.title(f"{name}_average_precision")
    plt.savefig(f"plots/{name}_ap.jpg")
    plt.close()
    x = list()
    y = list()
    for (key, values) in tail_acc.items():
        x.append(key.item())
        y.append(values.sum().item() / 20)
    plt.plot(x, y)
    plt.title(f"{name}_average_tail_accuracy")
    plt.xlabel("threshold")
    plt.savefig(f"plots/{name}_tail_acc.jpg")
    plt.close()

# ========================================================= #


def plot_tailacc():
    # Tail accuracy
    fivecrop_tailacc = torch.load(
        "saves/five_crop_tailacc.pth", map_location=torch.device('cpu'))
    randrot_tailacc = torch.load(
        "saves/rand_rotate_tailacc.pth", map_location=torch.device('cpu'))
    fc_x = [k.item() for k in fivecrop_tailacc.keys()]
    fc_y = [v.sum().item() / 20.0 for v in fivecrop_tailacc.values()]
    rr_x = [k.item() for k in randrot_tailacc.keys()]
    rr_y = [v.sum().item() / 20.0 for v in randrot_tailacc.values()]
    plt.plot(fc_x, fc_y)
    plt.plot(rr_x, rr_y)
    plt.xlabel("threshold")
    plt.legend(("FiveCrop", "RandRotCrop"))
    plt.savefig("plots/tail_acc.jpg")
    plt.close()


def plot_loss():
    # Val loss
    fivecrop_val_loss = np.load("saves/five_crop_val_loss.npy")
    randrot_val_loss = np.load("saves/rand_rotate_val_loss.npy")
    x = np.arange(len(fivecrop_val_loss))
    plt.plot(x, fivecrop_val_loss)
    plt.plot(x, randrot_val_loss)
    plt.xlabel("epoch")
    plt.legend(("FiveCrop", "RandRotCrop"))
    plt.savefig(f"plots/val_loss.jpg")
    plt.close()
    fivecrop_train_loss = np.load("saves/five_crop_train_loss.npy")
    randrot_train_loss = np.load("saves/rand_rotate_train_loss.npy")
    plt.plot(x, fivecrop_train_loss)
    plt.plot(x, randrot_train_loss)
    plt.xlabel("epoch")
    plt.legend(("FiveCrop", "RandRotCrop"))
    plt.savefig(f"plots/train_loss.jpg")
    plt.close()


def plot_accuracy():
    # Val loss
    fivecrop_acc = np.load(f"saves/five_crop_acc.npy")
    randrot_acc = np.load("saves/rand_rotate_acc.npy")
    x = np.arange(len(fivecrop_acc))
    plt.plot(x, fivecrop_acc)
    plt.plot(x, randrot_acc)
    plt.xlabel("epoch")
    plt.legend(("FiveCrop", "RandRotCrop"))
    plt.savefig(f"plots/accuracy.jpg")
    plt.close()


def plot_ap():
    # Average precision
    fivecrop_ap = np.load("saves/five_crop_ap.npy")
    randrot_ap = np.load("saves/rand_rotate_ap.npy")
    x = np.arange(len(fivecrop_ap))
    plt.plot(x, fivecrop_ap)
    plt.plot(x, randrot_ap)
    plt.xlabel("epoch")
    plt.title(f"average_precision")
    plt.legend(("FiveCrop", "RandRotCrop"))
    plt.savefig(f"plots/average_precision.jpg")
    plt.close()


if __name__ == "__main__":
    # plot("rand_rotate")
    # plot_tailacc()
    # plot_loss()
    # plot_accuracy()
    # plot_ap()
    top_confidence_list()
