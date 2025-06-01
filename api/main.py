import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from model.predict import predict_image
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        prediction = predict_image(image_bytes)

        response = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "report": {
                "verdict": prediction["verdict"],
                "ai": prediction["ai"],
                "human": prediction["human"],
                "generator": {
                    "midjourney": {"is_detected": False, "confidence": 0},
                    "dall_e": {"is_detected": False, "confidence": 0},
                    "stable_diffusion": {"is_detected": False, "confidence": 0},
                    "this_person_does_not_exist": {"is_detected": False, "confidence": 0},
                    "adobe_firefly": {"is_detected": False, "confidence": 0},
                    "flux": {"is_detected": False, "confidence": 0},
                    "four_o": {"is_detected": False, "confidence": 0}
                }
            },
            "facets": {
                "quality": {
                    "is_detected": bool(prediction["quality"])
            },
                "nsfw": {
                    "version": "1.0.0",
                    "is_detected": False
                }
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
