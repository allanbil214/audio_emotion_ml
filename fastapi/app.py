from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import librosa
import tensorflow as tf
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Load the saved model
MODEL_PATH = "C:/Users/Allan/Desktop/audify/fastapi/lstm_emotion_model.h5" 
model = tf.keras.models.load_model(MODEL_PATH)

# Define emotions
EMOTIONS = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'ps': 'pleasant_surprise',
    'sad': 'sad'
}

# Create a folder to save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Extract same features used during training
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

    return np.hstack([mfccs, chroma, contrast])

def preprocess_audio(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=1)  # Add time step dimension
    return features


@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    
    # Preprocess and predict
    input_data = preprocess_audio(file_location)
    prediction = model.predict(input_data)
    predicted_emotion = EMOTIONS[list(EMOTIONS.keys())[np.argmax(prediction)]]
    
    return {"emotion": predicted_emotion, "audio_file": file.filename}


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return {"error": "File not found"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
