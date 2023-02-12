import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from os.path import exists

app = Flask(__name__)

df_path = "Crop_recommendation.csv" if exists("Crop_recommendation.csv") else "https://github.com/ashllxyy/TRINIT-TrichyWaale-ML03/raw/main/Crop_recommendation.csv"
df = pd.read_csv(df_path, on_bad_lines='skip')
test_features = df.drop(columns=["label"])

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    prediction = model.predict(test_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='Crops grown should be {}'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
