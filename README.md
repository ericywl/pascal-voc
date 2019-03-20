# Pascal VOC Deep Learning Project

## Setup

Install 2 things before running the code.

-   PyTorch for all things deep-learning
-   Scikit-Learn only for the average precision measure, to avoid reinventing the wheel
-   Flask for the web server

```
pip install torch
pip install sklearn
pip install flask
```

## Instructions

### Training and validation

First, unpack the VOC tarfile before running `pascal.py`.

```
tar -xvf VOCtrainval_11-May-2012.tar
python pascal.py
```

`pascal.py` will run for 30 epochs and save the outputs in `saves/` and the best model weights into `weights/`.

### Web GUI

Run `python app.py` and head to `localhost:5000`.
