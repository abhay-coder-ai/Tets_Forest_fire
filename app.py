import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Flask app
application = Flask(__name__)
app = application

# -------------------------------
# Load scaler and model safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
model_path = os.path.join(BASE_DIR, "models", "ridge.pkl")

scaler = pickle.load(open(scaler_path, "rb"))
model = pickle.load(open(model_path, "rb"))

# -------------------------------
# Routes
# -------------------------------
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

        # Create DataFrame with feature names (NO WARNING)
        input_data = pd.DataFrame(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes]],
            columns=scaler.feature_names_in_
        )

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

    return render_template("index.html", prediction=prediction)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
