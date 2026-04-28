"""
TerraTrace - Real-Time Detection (sends to Cloud API)
======================================================
Records mic → sends audio to FastAPI on Render → API runs AI → pushes Firebase → dashboard updates

Install:
    pip install sounddevice requests plyer numpy scipy
"""

import numpy as np
import sounddevice as sd
import requests
import threading
import webbrowser
import tempfile
import os
import scipy.io.wavfile as wav
from datetime import datetime
from plyer import notification
import platform

# ─────────────────────────────────────────
# ⚙️  CONFIG — Update API_URL after deploying to Render!
# ─────────────────────────────────────────
API_URL = "https://YOUR-APP-NAME.onrender.com/detect"   # ← update after Render deploy

DEVICE_CONFIG = {
    "device_id": "TerraTrace-Node-01",
    "zone"     : "Zone A — North Entry",
    "lat"      : 19.2147,
    "lon"      : 72.9105,
}

SAMPLE_RATE      = 16000
DURATION         = 3        # seconds per chunk
COOLDOWN_SECONDS = 15       # seconds between alerts


# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
last_alert_time = None


# ─────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────

def record_audio():
    """Record DURATION seconds from mic, return as numpy array."""
    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.flatten()


def save_wav(audio_np):
    """Save numpy array to temp WAV file, return path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    # Convert float32 to int16 for WAV
    audio_int16 = (audio_np * 32767).astype(np.int16)
    wav.write(tmp.name, SAMPLE_RATE, audio_int16)
    return tmp.name


def send_to_api(wav_path):
    """Send audio file to FastAPI on Render, return result dict."""
    try:
        with open(wav_path, "rb") as f:
            response = requests.post(
                API_URL,
                files={"file": ("audio.wav", f, "audio/wav")},
                params={
                    "zone"     : DEVICE_CONFIG["zone"],
                    "device_id": DEVICE_CONFIG["device_id"],
                    "lat"      : DEVICE_CONFIG["lat"],
                    "lon"      : DEVICE_CONFIG["lon"],
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


def play_alarm():
    """Cross-platform alarm."""
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


def trigger_local_alert(result):
    """Show local alerts on laptop when API says threat."""
    zone   = result.get("zone", DEVICE_CONFIG["zone"])
    threat = result.get("threat_type", "Unknown")
    conf   = result.get("confidence", 0)
    lat    = DEVICE_CONFIG["lat"]
    lon    = DEVICE_CONFIG["lon"]
    now    = datetime.now().strftime("%H:%M:%S")

    # Terminal box
    print(f"""
╔══════════════════════════════════════╗
║       🚨 TERRATTRACE ALERT 🌿        ║
╠══════════════════════════════════════╣
║  Zone    : {zone:<27}║
║  Threat  : {threat:<27}║
║  Conf    : {f"{conf}%":<27}║
║  Time    : {now:<27}║
╚══════════════════════════════════════╝
    """)

    # Desktop popup
    try:
        notification.notify(
            title   ="🚨 TerraTrace ALERT",
            message =f"Threat: {threat}\nZone: {zone}\nConfidence: {conf}%",
            app_name="TerraTrace",
            timeout =10
        )
    except:
        pass

    # Alarm + Maps
    threading.Thread(target=play_alarm, daemon=True).start()
    webbrowser.open(f"https://maps.google.com/?q={lat},{lon}")


def handle_result(result):
    """Check cooldown then trigger alert."""
    global last_alert_time

    if not result:
        return

    is_threat  = result.get("is_threat", False)
    alert_sent = result.get("alert_sent", False)
    confidence = result.get("confidence", 0)
    threat     = result.get("threat_type", "Safe")

    if is_threat and alert_sent:
        now = datetime.now()
        if last_alert_time:
            elapsed = (now - last_alert_time).total_seconds()
            if elapsed < COOLDOWN_SECONDS:
                print(f"   ⏳ Cooldown: {COOLDOWN_SECONDS - elapsed:.0f}s left")
                return
        last_alert_time = now
        threading.Thread(
            target=trigger_local_alert,
            args=(result,),
            daemon=True
        ).start()
    else:
        print(f"✅ SAFE ({confidence}%)")


# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
def detection_loop():
    print("=" * 50)
    print("🎙️  TerraTrace LIVE — Sending to Cloud AI...")
    print(f"   API      : {API_URL}")
    print(f"   Zone     : {DEVICE_CONFIG['zone']}")
    print(f"   Location : {DEVICE_CONFIG['lat']}, {DEVICE_CONFIG['lon']}")
    print("=" * 50)
    print("   Press Ctrl+C to stop\n")

    while True:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 🎤 Recording ({DURATION}s)...",
            end=" ", flush=True
        )

        # 1. Record
        audio    = record_audio()
        wav_path = save_wav(audio)

        # 2. Send to API
        print("📡 Sending to Cloud...", end=" ", flush=True)
        result = send_to_api(wav_path)

        # 3. Clean up temp file
        try:
            os.unlink(wav_path)
        except:
            pass

        # 4. Handle result
        if result:
            threat = result.get("threat_type", "?")
            conf   = result.get("confidence", 0)
            print(f"→ {threat} ({conf}%)")
            handle_result(result)
        else:
            print("→ No response from API")


if __name__ == "__main__":
    try:
        detection_loop()
    except KeyboardInterrupt:
        print("\n\n🛑 TerraTrace stopped.")
