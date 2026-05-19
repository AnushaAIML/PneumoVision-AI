from fastapi import FastAPI, UploadFile, File

import cv2
import numpy as np
import traceback
import base64

from src.inference import predict


app = FastAPI()


@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        np_arr = np.frombuffer(
            contents,
            np.uint8
        )

        image = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        pred, conf, overlay = predict(image)

        # Convert overlay image to base64
        _, buffer = cv2.imencode(
            ".jpg",
            overlay
        )

        overlay_base64 = base64.b64encode(
            buffer
        ).decode("utf-8")

        return {
            "prediction":
                "PNEUMONIA"
                if pred == 1
                else "NORMAL",

            "confidence": float(conf),

            "heatmap": overlay_base64
        }

    except Exception as e:

        traceback.print_exc()

        return {
            "error": str(e)
        }