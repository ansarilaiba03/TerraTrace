"""
TerraTrace - Model Training Script
====================================
Uses YOUR OWN dataset (ambient/chainsaw/gunshot/vehicle folders)

Folder structure expected:
    TERRATRACE/
    ├── dataset/
    │   ├── ambient/     ← safe sounds
    │   ├── chainsaw/    ← threat
    │   ├── gunshot/     ← threat
    │   └── vehicle/     ← threat
    ├── train_model.py   ← this file
    └── realtime_detect.py

Install dependencies:
    pip install tensorflow tensorflow-hub numpy scikit-learn librosa soundfile
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa
import pickle

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATASET_PATH  = "dataset"       # folder containing ambient/chainsaw/gunshot/vehicle
SAMPLE_RATE   = 16000
DURATION      = 3               # seconds per clip
MODEL_SAVE    = "terrattrace_model.h5"
ENCODER_SAVE  = "label_encoder.pkl"

LABEL_MAP = {
    "chainsaw" : "threat",
    "gunshot"  : "threat",
    "vehicle"  : "threat",
    "ambient"  : "safe",
}


# ─────────────────────────────────────────
# STEP 1 — Load YAMNet
# ─────────────────────────────────────────
print("\n[1/4] Loading YAMNet from TensorFlow Hub...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("      ✅ YAMNet loaded!")


# ─────────────────────────────────────────
# STEP 2 — Audio Utilities
# ─────────────────────────────────────────
def load_audio(path, sr=SAMPLE_RATE, duration=DURATION):
    """Load audio file and normalize to fixed length."""
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    target_len = sr * duration
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio.astype(np.float32)


def extract_embedding(audio_np):
    """Run YAMNet and return averaged 1024-dim embedding."""
    scores, embeddings, _ = yamnet_model(audio_np)
    return np.mean(embeddings.numpy(), axis=0)


def augment_audio(audio):
    """
    Simple augmentation to increase dataset size.
    Returns list of augmented versions.
    """
    augmented = [audio]  # original

    # Add background noise
    noise = audio + 0.005 * np.random.randn(len(audio)).astype(np.float32)
    augmented.append(noise)

    # Pitch shift up slightly
    try:
        pitched = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=1)
        augmented.append(pitched.astype(np.float32))
    except:
        pass

    # Time stretch slightly
    try:
        stretched = librosa.effects.time_stretch(audio, rate=0.9)
        stretched = stretched[:SAMPLE_RATE * DURATION]
        if len(stretched) < SAMPLE_RATE * DURATION:
            stretched = np.pad(stretched, (0, SAMPLE_RATE * DURATION - len(stretched)))
        augmented.append(stretched.astype(np.float32))
    except:
        pass

    return augmented


# ─────────────────────────────────────────
# STEP 3 — Build Dataset from YOUR folders
# ─────────────────────────────────────────
print("\n[2/4] Loading your dataset...")

X = []
y = []

for folder_name, label in LABEL_MAP.items():
    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.exists(folder_path):
        print(f"      ⚠️  Folder not found: {folder_path} — skipping")
        continue

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.m4a'))
    ]

    print(f"\n      📂 {folder_name}/ ({label}) — {len(files)} files found")

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            audio = load_audio(filepath)

            # Augment data to get more samples
            versions = augment_audio(audio)

            for version in versions:
                emb = extract_embedding(version)
                X.append(emb)
                y.append(label)

            print(f"         ✅ {filename} → {len(versions)} samples")

        except Exception as e:
            print(f"         ⚠️  Skipped {filename}: {e}")

# Add synthetic safe samples (silence + noise)
print("\n      Adding synthetic safe (silence/noise) samples...")
for _ in range(20):
    silence = np.zeros(SAMPLE_RATE * DURATION, dtype=np.float32)
    X.append(extract_embedding(silence))
    y.append("safe")

for _ in range(20):
    noise = np.random.normal(0, 0.01, SAMPLE_RATE * DURATION).astype(np.float32)
    X.append(extract_embedding(noise))
    y.append("safe")

X = np.array(X)
y = np.array(y)

print(f"\n      ✅ Final dataset: {len(X)} total samples")
print(f"         🔴 Threats : {np.sum(y == 'threat')}")
print(f"         🟢 Safe    : {np.sum(y == 'safe')}")

if len(X) == 0:
    print("\n❌ No audio files found! Check your dataset folder structure.")
    exit()


# ─────────────────────────────────────────
# STEP 4 — Train Classifier
# ─────────────────────────────────────────
print("\n[3/4] Training classifier...")

# Encode labels: safe=0, threat=1
le = LabelEncoder()
y_encoded = le.fit_transform(y)

with open(ENCODER_SAVE, "wb") as f:
    pickle.dump(le, f)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

# Build classifier on top of YAMNet embeddings (1024-dim input)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")   # 0=safe, 1=threat
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            patience=8,
            restore_best_weights=True,
            monitor="val_accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=4,
            factor=0.5,
            monitor="val_loss"
        )
    ]
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n      ✅ Test Accuracy : {acc*100:.1f}%")
print(f"      ✅ Test Loss     : {loss:.4f}")

# Save
model.save(MODEL_SAVE)
print(f"\n[4/4] ✅ Model saved  → {MODEL_SAVE}")
print(f"      ✅ Encoder saved → {ENCODER_SAVE}")
print("\n🎉 Training complete! Now run: python realtime_detect.py")
