"""
TerraTrace - FastAPI Backend
==============================
Receives audio from realtime_detect.py
→ Runs AI model
→ Pushes alert to Firebase
→ Returns prediction

Deploy on Render (free)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import pickle
import tempfile
import librosa
import os

# ─────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────
app = FastAPI(
    title="TerraTrace API",
    description="Acoustic forest threat detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# LOAD MODELS ON STARTUP
# ─────────────────────────────────────────
print("🌿 TerraTrace API Starting...")

# Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
print("✅ Firebase connected!")

# YAMNet
print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("✅ YAMNet loaded!")

# Classifier
print("Loading classifier...")
classifier = tf.keras.models.load_model("terrattrace_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
print("✅ Classifier loaded!")

# YAMNet threat class map
YAMNET_THREAT_MAP = {
    314: "Chainsaw 🪚",
    427: "Gunshot 🔫",
    300: "Vehicle Engine 🚛",
    388: "Axe 🪓",
    289: "Wood Chopping 🪓",
    132: "Explosion 💥",
}

SAMPLE_RATE = 16000
DURATION    = 3


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    target   = SAMPLE_RATE * DURATION
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio[:target].astype(np.float32)


def extract_embedding(audio_np):
    scores, embeddings, _ = yamnet_model(audio_np)
    mean_scores  = np.mean(scores.numpy(), axis=0)
    top_class    = int(np.argmax(mean_scores))
    mean_emb     = np.mean(embeddings.numpy(), axis=0)
    return mean_emb, top_class


def predict(audio_np):
    embedding, top_class = extract_embedding(audio_np)
    emb_input  = np.expand_dims(embedding, axis=0)
    prediction = float(classifier.predict(emb_input, verbose=0)[0][0])
    is_threat  = prediction > 0.5
    confidence = prediction if is_threat else 1 - prediction
    threat_name = YAMNET_THREAT_MAP.get(top_class, "Unknown Threat ⚠️")
    return is_threat, confidence, threat_name


def push_firebase(zone, device_id, lat, lon, threat_name, confidence):
    db.collection("alerts").add({
        "zone"      : zone,
        "device_id" : device_id,
        "type"      : threat_name,
        "confidence": int(confidence * 100),
        "lat"       : lat,
        "lon"       : lon,
        "time"      : firestore.SERVER_TIMESTAMP,
    })


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status" : "🌿 TerraTrace API is live",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/detect")
async def detect(
    file     : UploadFile = File(...),
    zone     : str = "Zone A — North Entry",
    device_id: str = "TerraTrace-Node-01",
    lat      : float = 19.2147,
    lon      : float = 72.9105,
):
    """
    Receives audio file from edge device.
    Runs AI model → returns prediction → pushes to Firebase if threat.
    """
    # Validate file type
    if not file.filename.endswith(('.wav', '.mp3', '.ogg', '.flac')):
        raise HTTPException(400, "Only audio files accepted (.wav .mp3 .ogg .flac)")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Load + predict
        audio     = load_audio(tmp_path)
        is_threat, confidence, threat_name = predict(audio)

        result = {
            "is_threat"  : is_threat,
            "confidence" : round(confidence * 100, 1),
            "threat_type": threat_name if is_threat else "Safe ✅",
            "zone"       : zone,
            "device_id"  : device_id,
            "timestamp"  : datetime.now().isoformat(),
        }

        # Push to Firebase only if threat detected
        if is_threat and confidence >= 0.60:
            push_firebase(zone, device_id, lat, lon, threat_name, confidence)
            result["alert_sent"] = True
            print(f"🚨 THREAT: {threat_name} ({confidence*100:.1f}%) @ {zone}")
        else:
            result["alert_sent"] = False
            print(f"✅ Safe ({confidence*100:.1f}%)")

        return result

    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

    finally:
        os.unlink(tmp_path)  # clean up temp file
