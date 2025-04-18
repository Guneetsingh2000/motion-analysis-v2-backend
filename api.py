from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64
import numpy as np
import cv2
import tempfile
import os

app = FastAPI()

class VideoRequest(BaseModel):
    video_data: str
    option: int
    blur: int
    hist_eq: bool
    canny: bool

@app.get("/")
def root():
    return {"message": "Backend is running!"}

@app.post("/process")
def analyze_motion(req: VideoRequest):
    try:
        video_bytes = base64.b64decode(req.video_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        result = process_video(temp_path, req.option, req.blur, req.hist_eq, req.canny)
        os.remove(temp_path)

        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

def encode_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def preprocess(frame, blur=5, hist_eq=True, canny=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    if hist_eq:
        gray = cv2.equalizeHist(gray)
    if canny:
        gray = cv2.Canny(gray, 50, 150)
    return gray

def process_video(video_path, option, blur, hist_eq, canny):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise ValueError("Could not read video frame.")

    height, width, _ = frame.shape
    vis = frame.copy()
    speed = 0

    gray = preprocess(frame, blur, hist_eq, canny)

    if option == 1:
        cv2.rectangle(vis, (width//3, height//3), (width//3+50, height//3+50), (0, 255, 0), 2)
        speed = 12.5
    elif option == 2:
        corners = cv2.goodFeaturesToTrack(gray, 5, 0.01, 10)
        if corners is not None:
            for pt in corners:
                x, y = pt.ravel()
                cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), -1)
        speed = 22.0
    elif option == 3:
        cv2.putText(vis, "Object: bike (demo)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.rectangle(vis, (width//4, height//4), (width//4+100, height//4+100), (255,0,0), 2)
        speed = 35.7

    return {
        "max_speed": speed,
        "avg_speed": speed / 2,
        "min_speed": speed / 4,
        "overlay_frame": encode_image(vis)
    }

    
