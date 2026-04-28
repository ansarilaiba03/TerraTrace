"""
TerraTrace - FastAPI Backend (Lightweight)
==========================================
Laptop runs AI model locally → sends RESULT to this API
This API → saves alert to Firebase → dashboard updates live
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
    firebase_key = json.loads(os.environ["FIREBASE_KEY"])
    cred = credentials.Certificate(firebase_key)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase connected!")
except Exception as e:
    print(f"Firebase error: {e}")
    db = None


class AlertRequest(BaseModel):
    device_id  : str
    zone       : str
    lat        : float
    lon        : float
    threat_type: str
    confidence : float
    is_threat  : bool


@app.get("/")
def root():
    return {"status": "TerraTrace API is live", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "firebase": "connected" if db else "error",
        "timestamp": datetime.now().isoformat()
    }


# 🔴 MAIN FIX: clean, predictable response + Firebase write only when needed
@app.post("/alert")
def receive_alert(data: AlertRequest):
    if not db:
        raise HTTPException(500, "Firebase not connected")

    try:
        # If NOT a threat → do nothing but return clean response
        if not data.is_threat:
            return {
                "is_threat": False,
                "confidence": data.confidence,
                "threat_type": "Safe",
                "saved": False
            }

        # If threat → save to Firebase
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

        return {
            "is_threat": True,
            "confidence": data.confidence,
            "threat_type": data.threat_type,
            "saved": True
        }

    except Exception as e:
        raise HTTPException(500, f"Firebase save failed: {str(e)}")


@app.get("/alerts")
def get_alerts():
    if not db:
        raise HTTPException(500, "Firebase not connected")

    try:
        docs = db.collection("alerts") \
            .order_by("time", direction=firestore.Query.DESCENDING) \
            .limit(20) \
            .stream()

        alerts = []
        for doc in docs:
            d = doc.to_dict()
            d["id"] = doc.id
            alerts.append(d)

        return {"alerts": alerts}

    except Exception as e:
        raise HTTPException(500, str(e))