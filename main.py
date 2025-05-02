import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import cv2
import threading
from flask import Flask, Response, jsonify, render_template
import time

# Flask app setup
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# Load YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # chỉ nhận >=40% độ tin cậy

# Nếu có GPU thì chạy trên GPU
if torch.cuda.is_available():
    model.to('cuda')

# Chỉ lấy các class xe
vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'person']


def detect_vehicles():
    global output_frame

    # Kết nối camera (0 = webcam; hoặc ESP32-CAM IP)
    # cap = cv2.VideoCapture('http://192.168.1.61:80/stream')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to connect to camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize ảnh nhỏ lại
        frame = cv2.resize(frame, (640, 480))

        # YOLO nhận diện
        results = model(frame, size=640)

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
            output_frame = frame.copy()

        # Delay ngắn để giảm tải CPU
        time.sleep(0.01)

    cap.release()


@app.route('/video_feed')
def video_feed():
    """Stream video với kết quả phát hiện"""

    def generate():
        global output_frame
        while True:
            with lock:
                if output_frame is None:
                    continue

                # Mã hóa frame thành JPEG
                (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
                if not flag:
                    continue

                # Trả về frame dạng multipart response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encoded_image) +
                       b'\r\n')

            # Delay ngắn
            time.sleep(0.03)

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/')
def index():
    """Trang chủ hiển thị camera fullscreen"""
    return render_template('index.html')


@app.route('/api/info')
def api_info():
    """API thông tin dạng JSON"""
    return jsonify({
        "status": "running",
        "endpoints": {
            "video_feed": "/video_feed",
            "info": "/api/info"
        }
    })


if __name__ == '__main__':
    # Khởi động thread phát hiện xe
    t = threading.Thread(target=detect_vehicles)
    t.daemon = True
    t.start()

    # Khởi động Flask server
    print("Starting server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
