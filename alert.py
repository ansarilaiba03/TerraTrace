import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import random
import time

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

zones = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"]
threats = ["Chainsaw", "Gunshot", "Vehicle", "Human Activity"]

while True:
    alert = {
        "zone": random.choice(zones),
        "type": random.choice(threats),
        "confidence": random.randint(70, 99),
        "time": datetime.now()
    }

    db.collection("alerts").add(alert)

    print("Alert sent:", alert)

    time.sleep(5)
    