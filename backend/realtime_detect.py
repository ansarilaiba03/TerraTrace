"""
TerraTrace - Real-Time Detection + Firebase Alert
===================================================
Run AFTER train_model.py has finished.
Listens to laptop mic → detects threat → pushes to Firebase → dashboard updates live.

Install:
    pip install tensorflow tensorflow-hub numpy sounddevice librosa firebase-admin plyer
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
import threading
import pickle
import webbrowser
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from plyer import notification
import platform

# ─────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────
DEVICE_CONFIG = {
    "device_id" : "TerraTrace-Node-01",
    "zone"      : "Zone A — North Entry",
    "lat"       : 19.2147,      # Sanjay Gandhi National Park coords
    "lon"       : 72.9105,
}

SAMPLE_RATE       = 16000
DURATION          = 3         # seconds per audio chunk
CONFIDENCE_THRESH = 0.60      # trigger alert only above this
COOLDOWN_SECONDS  = 15        # seconds between alerts (avoid spam)

# YAMNet threat class indices → human-readable names
YAMNET_THREAT_MAP = {
    314: "Chainsaw 🪚",
    427: "Gunshot 🔫",
    300: "Vehicle Engine 🚛",
    388: "Axe 🪓",
    289: "Wood Chopping 🪓",
    132: "Explosion 💥",
}


# ─────────────────────────────────────────
# FIREBASE INIT
# ─────────────────────────────────────────
print("\n🌿 TerraTrace Starting Up...")
print("   Connecting to Firebase...")
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
print("   ✅ Firebase connected!")


# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
print("   Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

print("   Loading trained classifier...")
classifier = tf.keras.models.load_model("terrattrace_model.h5")

print("   Loading label encoder...")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("   ✅ All models loaded!\n")


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
last_alert_time = None


# ─────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────

def extract_embedding(audio_np):
    """Get YAMNet embedding + top class."""
    scores, embeddings, _ = yamnet_model(audio_np)
    mean_scores   = np.mean(scores.numpy(), axis=0)
    top_class_id  = int(np.argmax(mean_scores))
    mean_emb      = np.mean(embeddings.numpy(), axis=0)
    return mean_emb, top_class_id, float(mean_scores[top_class_id])


def predict_threat(audio_np):
    """
    Returns:
        is_threat   : bool
        confidence  : float 0.0-1.0
        threat_name : str
    """
    embedding, top_yamnet_class, _ = extract_embedding(audio_np)

    emb_input  = np.expand_dims(embedding, axis=0)
    prediction = float(classifier.predict(emb_input, verbose=0)[0][0])

    is_threat  = prediction > 0.5
    confidence = prediction if is_threat else 1 - prediction

    # Map YAMNet class to name if it's a known threat
    threat_name = YAMNET_THREAT_MAP.get(top_yamnet_class, "Unknown Threat ⚠️")

    return is_threat, confidence, threat_name


def push_to_firebase(threat_name, confidence):
    """Push alert document to Firestore."""
    try:
        db.collection("alerts").add({
            "zone"      : DEVICE_CONFIG["zone"],
            "type"      : threat_name,
            "confidence": int(confidence * 100),
            "lat"       : DEVICE_CONFIG["lat"],
            "lon"       : DEVICE_CONFIG["lon"],
            "device_id" : DEVICE_CONFIG["device_id"],
            "time"      : firestore.SERVER_TIMESTAMP,
        })
        print("   ✅ Pushed to Firebase!")
    except Exception as e:
        print(f"   ❌ Firebase error: {e}")


def play_alarm():
    """Cross-platform alarm beep."""
    try:
        if platform.system() == "Windows":
            import winsound
            for _ in range(3):
                winsound.Beep(1000, 400)
        elif platform.system() == "Darwin":  # Mac
            os.system('say "Alert! Threat Detected!"')
        else:  # Linux
            os.system('echo -e "\a"')
    except:
        pass


def send_alert(threat_name, confidence):
    """Full alert pipeline: Firebase + notification + maps + terminal."""
    lat  = DEVICE_CONFIG["lat"]
    lon  = DEVICE_CONFIG["lon"]
    zone = DEVICE_CONFIG["zone"]
    now  = datetime.now().strftime("%H:%M:%S")

    # 1. Push to Firebase → dashboard updates live
    push_to_firebase(threat_name, confidence)

    # 2. Desktop popup notification
    try:
        notification.notify(
            title   = "🚨 TerraTrace ALERT",
            message = f"Threat: {threat_name}\nZone: {zone}\nConfidence: {confidence*100:.0f}%",
            app_name= "TerraTrace",
            timeout = 10
        )
    except Exception as e:
        print(f"   ⚠️  Notification error: {e}")

    # 3. Alarm beep
    threading.Thread(target=play_alarm, daemon=True).start()

    # 4. Open Google Maps in browser
    webbrowser.open(f"https://maps.google.com/?q={lat},{lon}")

    # 5. Terminal alert box
    print(f"""
╔══════════════════════════════════════╗
║       🚨 TERRATTRACE ALERT 🌿        ║
╠══════════════════════════════════════╣
║  Zone    : {zone:<27}║
║  Threat  : {threat_name:<27}║
║  Conf    : {f"{confidence*100:.0f}%":<27}║
║  Time    : {now:<27}║
║  Maps    : maps.google.com/?q={lat},{lon}  
╚══════════════════════════════════════╝
    """)


def handle_threat(threat_name, confidence):
    """Cooldown check then trigger alert."""
    global last_alert_time

    now = datetime.now()
    if last_alert_time:
        elapsed = (now - last_alert_time).total_seconds()
        if elapsed < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - elapsed
            print(f"   ⏳ Cooldown: {remaining:.0f}s left — skipping alert")
            return

    last_alert_time = now
    threading.Thread(
        target=send_alert,
        args=(threat_name, confidence),
        daemon=True
    ).start()


# ─────────────────────────────────────────
# MAIN DETECTION LOOP
# ─────────────────────────────────────────
def detection_loop():
    print("=" * 50)
    print("🎙️  TerraTrace LIVE — Listening to mic...")
    print(f"   Device   : {DEVICE_CONFIG['device_id']}")
    print(f"   Zone     : {DEVICE_CONFIG['zone']}")
    print(f"   Location : {DEVICE_CONFIG['lat']}, {DEVICE_CONFIG['lon']}")
    print(f"   Threshold: {CONFIDENCE_THRESH*100:.0f}% confidence")
    print(f"   Cooldown : {COOLDOWN_SECONDS}s between alerts")
    print("=" * 50)
    print("   Press Ctrl+C to stop\n")

    chunk_samples = int(SAMPLE_RATE * DURATION)

    while True:
        # Record from mic
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🎤 Listening ({DURATION}s)...",
            end=" ", flush=True
        )

        audio = sd.rec(
            chunk_samples,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        audio_flat = audio.flatten()

        # Predict
        is_threat, confidence, threat_name = predict_threat(audio_flat)

        # Act
        if is_threat and confidence >= CONFIDENCE_THRESH:
            print(f"\n🚨 THREAT! {threat_name} ({confidence*100:.1f}%)")
            handle_threat(threat_name, confidence)
        else:
            label = "THREAT(low conf)" if is_threat else "SAFE ✅"
            print(f"{label} ({confidence*100:.1f}%)")


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    try:
        detection_loop()
    except KeyboardInterrupt:
        print("\n\n🛑 TerraTrace stopped.")
