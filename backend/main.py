"""
TerraTrace - FastAPI Backend (Lightweight)
==========================================
Laptop runs AI model locally → sends RESULT to this API
This API → saves alert to Firebase → dashboard updates live

No TensorFlow needed here — runs on Render free tier easily.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os
import json

app = FastAPI(
    title="TerraTrace API",
    description="Forest threat alert API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase
print("Starting TerraTrace API...")
try:
    firebase_key = os.environ.get("FIREBASE_KEY")
    if firebase_key:
        key_dict = json.loads(firebase_key)
        cred = credentials.Certificate(key_dict)
    else:
        cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase connected!")
except Exception as e:
    print(f"Firebase error: {e}")
    db = None

class AlertRequest(BaseModel):
    device_id  : str   = "TerraTrace-Node-01"
    zone       : str   = "Zone A - North Entry"
    lat        : float = 19.2147
    lon        : float = 72.9105
    threat_type: str
    confidence : float
    is_threat  : bool

@app.get("/")
def root():
    return {"status": "TerraTrace API is live", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "firebase": "connected" if db else "error", "timestamp": datetime.now().isoformat()}

@app.post("/alert")
def receive_alert(data: AlertRequest):
    if not db:
        raise HTTPException(500, "Firebase not connected")
    if not data.is_threat:
        return {"status": "safe", "saved": False}
    try:
        db.collection("alerts").add({
            "device_id" : data.device_id,
            "zone"      : data.zone,
            "lat"       : data.lat,
            "lon"       : data.lon,
            "type"      : data.threat_type,
            "confidence": int(data.confidence),
            "time"      : firestore.SERVER_TIMESTAMP,
        })
        print(f"Alert saved: {data.threat_type} ({data.confidence}%) @ {data.zone}")
        return {"status": "alert_saved", "saved": True, "threat": data.threat_type}
    except Exception as e:
        raise HTTPException(500, f"Firebase save failed: {str(e)}")

@app.get("/alerts")
def get_alerts():
    if not db:
        raise HTTPException(500, "Firebase not connected")
    try:
        docs = db.collection("alerts").order_by("time", direction=firestore.Query.DESCENDING).limit(20).stream()
        alerts = []
        for doc in docs:
            d = doc.to_dict()
            d["id"] = doc.id
            alerts.append(d)
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(500, str(e))
