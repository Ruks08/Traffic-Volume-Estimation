import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Get inputs directly as float values
        inputs = request.form.to_dict()

        # Convert values to floats (they are already numeric in the HTML form)
        numeric_values = [
            float(inputs['holiday']),
            float(inputs['temp']),
            float(inputs['rain']),
            float(inputs['snow']),
            float(inputs['weather']),
            float(inputs['day']),
            float(inputs['month']),
            float(inputs['year']),
            float(inputs['hours']),
            float(inputs['minutes']),
            float(inputs['seconds'])
        ]

        # Create DataFrame
        columns = ['holiday', 'temp', 'rain', 'snow', 'weather',
                   'day', 'month', 'year', 'hours', 'minutes', 'seconds']
        df = pd.DataFrame([numeric_values], columns=columns)

        # Predict
        prediction = model.predict(df)[0]
        result = f"🚦 Estimated Traffic Volume: [{prediction:.2f}] units"



        return render_template("output.html", result=result)

    except Exception as e:
        return render_template("output.html", result=f"⚠️ Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
