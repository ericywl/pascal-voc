# Pascal VOC Deep Learning Project


## Setup

Install 2 things before running the code.
- PyTorch for all things deep-learning
- PyTorch TorchNet for the average precision measure, avoid reinventing the
    wheel
- Flask for the web GUI

```
pip install torch
pip install git+https://github.com/pytorch/tnt.git@master
pip install flask
```

## Instructions
First, unpack the VOC tarfile, then run the Python scripts.

```
tar -xvf VOCtrainval_11-May-2012.tar
python pascal.py
```
