# Pascal VOC Deep Learning Project

## Setup

Install 2 things before running the code.

-   PyTorch for all things deep-learning
-   Scikit-Learn only for the average precision measure, to avoid reinventing
        the wheel
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
tar -xvf ./VOCtrainval_11-May-2012.tar
python pascal.py
```

`pascal.py` will run for `MAX_EPOCHS` epochs, which we set to 40.
Then it will save the outputs in `saves/` and the
best model weights in `weights/` if `SAVE_OUTPUTS` is enabled.

### Web GUI

To get the full experience of the web GUI (ranked image preview),
unpack the VOC tarfile and copy the `JPEGImages` folder into `static/images`
like so:

```
tar -xvf ./VOCtrainval_11-May-2012.tar
cp ./VOCdevkit/VOC2012/JPEGImages/* ./static/images/
```

Then, run `python app.py` and head to `localhost:5000` on a browser.
