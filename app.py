import base64
import re
import io
import json
from flask import Flask, render_template, request, jsonify

from pascal import PascalClassifier


app = Flask(__name__)
pc = PascalClassifier(weights_path="weights/five_crop_weights.pth")
classes = list(pc.CLASS_OCC_DICT.keys())

with open("saves/ranks.json") as f:
    ranks_dict = json.load(f)
    ranks_dict = {int(k) + 1: v for k, v in ranks_dict.items()}


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    if request.method == "POST":
        # Request data is not in JSON format
        if not request.content_type == "application/json":
            return "", "400 REQUEST DATA NOT JSON"
        img_data = re.sub("^data:image/.+;base64,", "",
                          request.get_json()["data"])
        img_blob = base64.b64decode(img_data)
        output = pc.predict(io.BytesIO(img_blob))
        pred = output > 0.5
        indices = sorted(range(len(output)), reverse=True,
                         key=lambda i: output[i].item())[:pred.sum().item()]
        pred_confidence = {classes[i]: output[i].item() for i in indices}
        return jsonify(pred_confidence)
    # Invalid HTTP request method
    return "", "400 INVALID REQUEST METHOD"


@app.route("/ranks")
def ranks():
    return render_template("ranks.html", data=ranks_dict)


@app.route("/ranks/<class_name>")
def class_ranks(class_name):
    class_index = classes.index(class_name) + 1
    if class_index < 1 or class_index > 20:
        return "", "400 INVALID REQUEST"
    c_ranks = ranks_dict[class_index]
    return render_template("class.html", class_name=class_name, data=c_ranks)


if __name__ == "__main__":
    app.run()
