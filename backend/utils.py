import cv2
import numpy as np
import base64

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

def process_video(frame, option, blur, hist_eq, canny):
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
