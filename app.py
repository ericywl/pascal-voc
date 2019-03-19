import base64
import re
import io
from flask import Flask, render_template, request, jsonify

from pascal import PascalClassifier


app = Flask(__name__)
pc = PascalClassifier(weights_path="weights/five_crop_weights.pth")


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")
    if request.method == "POST":
        # TODO: add image to local and do prediction
        if not request.content_type == "application/json":
            return "", "400 REQUEST DATA NOT JSON"
        img_data = re.sub("^data:image/.+;base64,", "",
                          request.get_json()["data"])
        img_blob = base64.b64decode(img_data)
        print(pc.predict(io.BytesIO(img_blob)))
        return jsonify({"test": "hello"})
    # Invalid HTTP request method
    return "", "400 INVALID REQUEST METHOD"


@app.route("/ranks")
def ranks():
    return render_template("ranks.html")


if __name__ == "__main__":
    app.run(debug=True)
