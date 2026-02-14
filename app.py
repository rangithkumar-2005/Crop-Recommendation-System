from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from main import load_data  # to use if you need data filtering later

app = Flask(__name__)

# Load model and data
model = joblib.load('crop_model.pkl')
df = load_data("Crop_recommendation.csv")  # optional, useful for filtering or analysis

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Explicitly fetch and convert form values in model's expected order
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_features = [[N, P, K, temperature, humidity, ph, rainfall]]

        # Make prediction
        prediction = model.predict(input_features)[0]
        crop_name = prediction.strip().lower()  # lowercase crop name to match filename

        # Render HTML with prediction and matching plots
        return render_template('index.html',
            prediction_text=f"✅ Recommended Crop: {prediction.capitalize()}",
            scatter_plot=f"plots/{crop_name}_scatter.png",
            bar_plot=f"plots/{crop_name}_bar.png",
            box_plot=f"plots/{crop_name}_box.png"
        )
    except Exception as e:
        return render_template('index.html',
            prediction_text=f"❌ Error: {str(e)}",
            scatter_plot=None,
            bar_plot=None,
            box_plot=None
        )

if __name__ == '__main__':
    app.run(debug=True)
