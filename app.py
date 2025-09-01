from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load model at startup
model = YOLO("best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
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
