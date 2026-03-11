from flask import Flask, render_template, request
from predict import predict_message

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():

    prediction = None
    probability = None

    if request.method == "POST":

        message = request.form["message"]

        prediction, probability = predict_message(message)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)