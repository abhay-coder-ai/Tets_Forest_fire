import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load scaler and model
scaler = pickle.load(open("models/scaler.pkl", "rb"))
model = pickle.load(open("models/ridge.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        Temperature = float(request.form["Temperature"])
        RH = float(request.form["RH"])
        Ws = float(request.form["Ws"])
        Rain = float(request.form["Rain"])
        FFMC = float(request.form["FFMC"])
        DMC = float(request.form["DMC"])
        ISI = float(request.form["ISI"])
        Classes = float(request.form["Classes"])

        # SAME order as scaler.feature_names_in_
        input_data = np.array([[Temperature, RH, Ws, Rain,
                                FFMC, DMC, ISI, Classes]])

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
