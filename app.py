from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def main():
    return render_template("index.html")
    
@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/ranks")
def ranks():
    return render_template("ranks.html")


if __name__ == "__main__":
    app.run(debug=True)
