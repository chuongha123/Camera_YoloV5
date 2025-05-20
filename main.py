import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import cv2
import threading
from flask import Flask, Response, jsonify, render_template
import time

# Flask app setup
app = Flask(__name__)
lock = threading.Lock()

# Cấu hình cameras
CAMERAS = {
    "bai1": {
        "name": "Bãi 1",
        "url": 0,  # 0 = webcam
        "output_frame": None
    },
    "bai2": {
        "name": "Bãi 2",
        "url": 1,  # 1 = second camera (or another index)
        "output_frame": None
    }
}

# Load YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # chỉ nhận >=40% độ tin cậy

# Nếu có GPU thì chạy trên GPU
if torch.cuda.is_available():
    model.to('cuda')

# Chỉ lấy các class xe
vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'person']

def detect_vehicles(camera_id):
    camera_config = CAMERAS[camera_id]
    
    # Kết nối camera
    cap = cv2.VideoCapture(camera_config["url"])

    if not cap.isOpened():
        print(f"Failed to connect to camera {camera_id}.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_id}.")
            time.sleep(1)  # Wait before retry
            continue

        # YOLO nhận diện
        results = model(frame)
        
        # Duyệt qua kết quả
        for *xyxy, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label in vehicle_classes:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Cập nhật output_frame có khóa để tránh race condition
        with lock:
            camera_config["output_frame"] = frame.copy()

    cap.release()

def gen_camera_stream(camera_id):
    camera_config = CAMERAS[camera_id]
    
    while True:
        # Lấy frame từ output_frame
        with lock:
            if camera_config["output_frame"] is None:
                continue
            frame = camera_config["output_frame"].copy()
        
        # Mã hóa frame thành JPEG
        (flag, encoded_image) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
            
        # Trả về frame dạng multipart response
        yield(b'--frame\r\n' 
              b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + 
              b'\r\n')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Stream video với kết quả phát hiện cho camera cụ thể"""
    if camera_id not in CAMERAS:
        return "Camera not found", 404
        
    return Response(gen_camera_stream(camera_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/')
def index():
    """Trang chủ hiển thị tất cả các camera"""
    return render_template('index.html', cameras=CAMERAS)

@app.route('/api/info')
def api_info():
    """API thông tin dạng JSON"""
    return jsonify({
        "status": "running",
        "cameras": list(CAMERAS.keys()),
        "endpoints": {
            "video_feeds": {camera_id: f"/video_feed/{camera_id}" for camera_id in CAMERAS},
            "info": "/api/info"
        }
    })

if __name__ == '__main__':
    # Khởi động thread phát hiện xe cho mỗi camera
    for camera_id in CAMERAS:
        t = threading.Thread(target=detect_vehicles, args=(camera_id,))
        t.daemon = True
        t.start()
    
    # Khởi động Flask server
    print("Starting server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
