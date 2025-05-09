from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import pandas as pd
import librosa
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Lazy-load variables
model = None
label_encoder = None
feature_columns = None

def load_resources():
    global model, label_encoder, feature_columns
    if model is None or label_encoder is None or feature_columns is None:
        model = joblib.load("backend/cry_model_5.pkl")
        label_encoder = joblib.load("backend/label_encoder.pkl")
        df = pd.read_csv("backend/processed_audio_features.csv")
        feature_columns = df.drop(columns=["cry_type"]).columns

# Feature extraction function (same as training)
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    fmin = max(20, sr * 0.01)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=fmin, n_bands=6)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    return np.hstack([mfcc_mean, chroma_mean, spectral_contrast_mean, zcr_mean])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_resources()  # Load models on-demand

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        features = extract_features(file_path).reshape(1, -1)
        features_df = pd.DataFrame(features, columns=feature_columns)
        prediction_encoded = model.predict(features_df)[0]
        predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]
        os.remove(file_path)  # Clean up uploaded file

        return jsonify({'prediction': predicted_label})
    except Exception as e:
        os.remove(file_path)  # Clean up on error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
