import matplotlib.pyplot as plt
import torch
import numpy as np

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
    plt.savefig(f"saves/{name}_val_loss.jpg")
    plt.close()
    # Train loss
    plt.plot(x, train_loss)
    plt.xlabel("epoch")
    plt.title(f"{name}_training_loss")
    plt.savefig(f"saves/{name}_train_loss.jpg")
    plt.close()
    # Accuracy
    plt.plot(x, accuracy)
    plt.xlabel("epoch")
    plt.title(f"{name}_accuracy")
    plt.savefig(f"saves/{name}_acc.jpg")
    plt.close()
    # Average precision
    plt.plot(x, ap)
    plt.xlabel("epoch")
    plt.title(f"{name}_average_precision")
    plt.savefig(f"saves/{name}_ap.jpg")
    plt.close()
    x = list()
    y = list()
    for (key, values) in tail_acc.items():
        x.append(key.item())
        y.append(values.sum().item() / 20)
    plt.plot(x, y)
    plt.title(f"{name}_average_tail_accuracy")
    plt.xlabel("threshold")
    plt.savefig(f"saves/{name}_tail_acc.jpg")
    plt.close()

# ========================================================= #


if __name__ == "__main__":
    plot("rand_rotate")
