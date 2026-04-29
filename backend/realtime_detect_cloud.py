"""
TerraTrace - Real-Time Detection (Local ML → Cloud API)
========================================================
Records mic → runs YAMNet + classifier locally → sends RESULT to FastAPI → Firebase → dashboard
"""

import numpy as np
import sounddevice as sd
import requests
import threading
import webbrowser
import tempfile
import os
import pickle
import scipy.io.wavfile as wav
from datetime import datetime
from plyer import notification
import platform
import tensorflow as tf
import tensorflow_hub as hub

# ─────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────
API_URL = "https://terratrace-bm16.onrender.com/alert"

DEVICE_CONFIG = {
    "device_id": "TerraTrace-Node-01",
    "zone"     : "Zone A — North Entry",
    "lat"      : 19.2147,
    "lon"      : 72.9105,
}

MODEL_PATH   = "backend/terrattrace_model.h5"
ENCODER_PATH = "backend/label_encoder.pkl"

SAMPLE_RATE      = 16000
DURATION         = 3
COOLDOWN_SECONDS = 15
THREAT_THRESHOLD = 0.7    # confidence above this = threat

LABEL_MAP = {
    "chainsaw" : "Chainsaw",
    "gunshot"  : "Gunshot",
    "vehicle"  : "Vehicle",
    "ambient"  : "Safe",
}

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
print("\n🌿 TerraTrace Starting Up...")
print("   Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("   ✅ YAMNet loaded!")

print("   Loading trained classifier...")
classifier = tf.keras.models.load_model(MODEL_PATH)
print("   ✅ Classifier loaded!")

print("   Loading label encoder...")
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)
print("   ✅ Label encoder loaded!")

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
last_alert_time = None


# ─────────────────────────────────────────
# AUDIO FUNCTIONS
# ─────────────────────────────────────────
def record_audio():
    """Record DURATION seconds from mic."""
    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.flatten()


def extract_embedding(audio_np):
    """Run YAMNet and return averaged 1024-dim embedding."""
    scores, embeddings, _ = yamnet_model(audio_np)
    return np.mean(embeddings.numpy(), axis=0)


def classify_audio(audio_np):
    """
    Run full pipeline: audio → YAMNet embedding → classifier.
    Returns (is_threat, threat_type, confidence_percent).
    """
    embedding = extract_embedding(audio_np)
    embedding = np.expand_dims(embedding, axis=0)  # shape (1, 1024)

    raw_score = classifier.predict(embedding, verbose=0)[0][0]  # 0=safe, 1=threat

    is_threat  = raw_score >= THREAT_THRESHOLD
    confidence = round(float(raw_score if is_threat else 1 - raw_score) * 100, 1)

    # Get the most likely YAMNet class for threat labelling
    scores, _, _ = yamnet_model(audio_np)
    top_class = np.argmax(np.mean(scores.numpy(), axis=0))

    # Map to a human label
    threat_type = "Unknown Threat"
    if is_threat:
        score_means = np.mean(scores.numpy(), axis=0)
        # Check known threat keywords in YAMNet class names (indices are stable)
        # Gunshot ~427, Chainsaw ~549, Vehicle/engine ~300-320
        gunshot_score  = score_means[427] if len(score_means) > 427 else 0
        chainsaw_score = score_means[549] if len(score_means) > 549 else 0
        vehicle_score  = max(score_means[300:320]) if len(score_means) > 320 else 0

        best = max(gunshot_score, chainsaw_score, vehicle_score)
        if best == gunshot_score:
            threat_type = "Gunshot"
        elif best == chainsaw_score:
            threat_type = "Chainsaw"
        elif best == vehicle_score:
            threat_type = "Vehicle"
    else:
        threat_type = "Safe"

    return is_threat, threat_type, confidence


# ─────────────────────────────────────────
# API FUNCTION
# ─────────────────────────────────────────
def send_to_api(is_threat, threat_type, confidence):
    """Send classification result as JSON to FastAPI."""
    try:
        response = requests.post(
            API_URL,
            json={
                "device_id"  : DEVICE_CONFIG["device_id"],
                "zone"       : DEVICE_CONFIG["zone"],
                "lat"        : DEVICE_CONFIG["lat"],
                "lon"        : DEVICE_CONFIG["lon"],
                "threat_type": threat_type,
                "confidence" : float(confidence),
                "is_threat"  : bool(is_threat),
            },
            timeout=30
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot reach API — check your Render URL")
        return None
    except Exception as e:
        print(f"   ❌ API error: {e}")
        return None


# ─────────────────────────────────────────
# ALERT FUNCTIONS
# ─────────────────────────────────────────
def play_alarm():
    """Cross-platform alarm beep."""
    try:
        if platform.system() == "Windows":
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 400)
        elif platform.system() == "Darwin":
            os.system('say "Threat Detected"')
        else:
            os.system('echo -e "\a"')
    except:
        pass


def trigger_local_alert(threat_type, confidence):
    """Show terminal box + desktop notification + alarm + maps."""
    now = datetime.now().strftime("%H:%M:%S")
    zone = DEVICE_CONFIG["zone"]
    lat  = DEVICE_CONFIG["lat"]
    lon  = DEVICE_CONFIG["lon"]

    print(f"""
╔══════════════════════════════════════╗
║       🚨 TERRATRACE ALERT 🌿         ║
╠══════════════════════════════════════╣
║  Zone    : {zone:<27}║
║  Threat  : {threat_type:<27}║
║  Conf    : {f"{confidence}%":<27}║
║  Time    : {now:<27}║
╚══════════════════════════════════════╝
    """)

    try:
        notification.notify(
            title   ="🚨 TerraTrace ALERT",
            message =f"Threat: {threat_type}\nZone: {zone}\nConfidence: {confidence}%",
            app_name="TerraTrace",
            timeout =10
        )
    except:
        pass

    threading.Thread(target=play_alarm, daemon=True).start()
    webbrowser.open(f"https://maps.google.com/?q={lat},{lon}")


def handle_result(is_threat, threat_type, confidence):
    """Check cooldown, trigger alert if needed."""
    global last_alert_time

    if is_threat:
        now = datetime.now()
        if last_alert_time:
            elapsed = (now - last_alert_time).total_seconds()
            if elapsed < COOLDOWN_SECONDS:
                print(f"   ⏳ Cooldown: {COOLDOWN_SECONDS - elapsed:.0f}s left")
                return
        last_alert_time = now
        threading.Thread(
            target=trigger_local_alert,
            args=(threat_type, confidence),
            daemon=True
        ).start()
    else:
        print(f"✅ SAFE ({confidence}%)")


# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
def detection_loop():
    print("\n" + "=" * 50)
    print("🎙️  TerraTrace LIVE — Local AI + Cloud Logging")
    print(f"   API      : {API_URL}")
    print(f"   Zone     : {DEVICE_CONFIG['zone']}")
    print(f"   Location : {DEVICE_CONFIG['lat']}, {DEVICE_CONFIG['lon']}")
    print(f"   Threshold: {THREAT_THRESHOLD * 100:.0f}% confidence")
    print("=" * 50)
    print("   Press Ctrl+C to stop\n")

    while True:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🎤 Recording ({DURATION}s)...",
            end=" ", flush=True
        )

        # 1. Record
        audio = record_audio()

        # 2. Classify locally
        print("🧠 Classifying...", end=" ", flush=True)
        is_threat, threat_type, confidence = classify_audio(audio)
        print(f"→ {threat_type} ({confidence}%)", end=" ", flush=True)

        # 3. Send result to API
        print("📡 Syncing...", end=" ", flush=True)
        api_response = send_to_api(is_threat, threat_type, confidence)

        if api_response:
            saved = api_response.get("saved", False)
            print("💾 Saved" if saved else "⏭️  Skipped")
        else:
            print("⚠️  API unreachable")

        # 4. Handle local alert
        handle_result(is_threat, threat_type, confidence)


if __name__ == "__main__":
    try:
        detection_loop()
    except KeyboardInterrupt:
        print("\n\n🛑 TerraTrace stopped.")