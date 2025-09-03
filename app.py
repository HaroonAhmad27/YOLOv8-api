from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI()

# Try loading model
MODEL_PATH = "best.pt"
model = None

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    print("⚠️ best.pt not found. Please upload or download at runtime.")

@app.get("/")
def root():
    return {"message": "API is live ✅"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={"error": "Model not loaded"}, status_code=500)

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    results = model.predict(image)

    response = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        response.append({
            "name": model.names[cls_id],
            "confidence": conf,
            "box": xyxy
        })

    return JSONResponse(content=response)
