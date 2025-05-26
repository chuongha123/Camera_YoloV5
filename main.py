import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import cv2
import threading
from flask import Flask, Response, jsonify, render_template
import time
import numpy as np
from queue import Queue

# Flask app setup
app = Flask(__name__)
lock = threading.Lock()

# Cấu hình cameras
CAMERAS = {
    "bai1": {
        "name": "Bãi 1",
        "url": 0,  # First webcam
        "output_frame": None,
        "last_results": None,
        "last_update_time": 0,
        "frame_queue": Queue(maxsize=10)  # Separate queue for bai1
    },
    "bai2": {
        "name": "Bãi 2",
        "url": 1,  # Second webcam
        "output_frame": None,
        "last_results": None,
        "last_update_time": 0,
        "frame_queue": Queue(maxsize=10)  # Separate queue for bai2
    },
}

# Load YOLOv5 pretrained model (sử dụng YOLOv5n với input size nhỏ hơn)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.4  # Chỉ nhận >=40% độ tin cậy
model.imgsz = 416  # Giảm kích thước input xuống 416x416

# Nếu có GPU thì chạy trên GPU
if torch.cuda.is_available():
    model.to('cuda')

# Chỉ lấy các class xe
vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'person']


# Hàm tiền xử lý frame
def preprocess_frame(frame, target_size=(416, 416)):
    orig_h, orig_w = frame.shape[:2]
    scale = min(target_size[0] / orig_w, target_size[1] / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame

    return canvas, (x_offset, y_offset, scale, orig_w, orig_h)


def capture_frames(camera_id):
    """Hàm capture frame từ camera và đưa vào queue riêng của camera"""
    camera_config = CAMERAS[camera_id]
    cap = cv2.VideoCapture(camera_config["url"])

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera {camera_id} resolution: {actual_width}x{actual_height}")

    if not cap.isOpened():
        print(f"Failed to connect to camera {camera_id}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_id}.")
            time.sleep(1)
            continue

        # Đưa frame vào queue riêng của camera
        if camera_config["frame_queue"].full():
            camera_config["frame_queue"].get()  # Loại bỏ frame cũ
        camera_config["frame_queue"].put((frame, time.time()))

    cap.release()


def process_frames(camera_id):
    """Hàm xử lý frame từ queue của camera với YOLOv5"""
    camera_config = CAMERAS[camera_id]
    frame_count = 0
    skip_frames = 30
    hold_time = 1.0

    while True:
        if camera_config["frame_queue"].empty():
            time.sleep(0.01)  # Đợi nếu queue rỗng
            continue

        frame, capture_time = camera_config["frame_queue"].get()
        processed_frame, (x_offset, y_offset, scale, orig_w, orig_h) = preprocess_frame(frame, target_size=(416, 416))

        current_time = time.time()

        # Run YOLO only for each frame skip_frames
        if frame_count % skip_frames == 0:
            try:
                results = model(processed_frame)
                with lock:
                    camera_config["last_results"] = results.xyxy[0]
                    camera_config["last_update_time"] = current_time
            except Exception as e:
                print(f"YOLO processing error for {camera_id}: {e}")
                continue

        # Vẽ bounding box từ kết quả gần nhất
        with lock:
            if (camera_config["last_results"] is not None and
                    current_time - camera_config["last_update_time"] <= hold_time):
                for *xyxy, conf, cls in camera_config["last_results"]:
                    label = model.names[int(cls)]
                    if label in vehicle_classes:
                        x1 = int((xyxy[0] - x_offset) / scale)
                        y1 = int((xyxy[1] - y_offset) / scale)
                        x2 = int((xyxy[2] - x_offset) / scale)
                        y2 = int((xyxy[3] - y_offset) / scale)
                        x1 = max(0, min(x1, orig_w - 1))
                        y1 = max(0, min(y1, orig_h - 1))
                        x2 = max(0, min(x2, orig_w - 1))
                        y2 = max(0, min(y2, orig_h - 1))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Cập nhật output_frame
        with lock:
            camera_config["output_frame"] = frame.copy()

        frame_count += 1


def gen_camera_stream(camera_id):
    """Stream video từ output_frame của camera"""
    camera_config = CAMERAS[camera_id]

    while True:
        with lock:
            if camera_config["output_frame"] is None:
                continue
            frame = camera_config["output_frame"].copy()

        # Mã hóa frame thành JPEG
        (flag, encoded_image) = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) +
               b'\r\n')


@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Stream video với kết quả phát hiện"""
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
    # Khởi động thread cho mỗi camera
    for camera_id in CAMERAS:
        # Thread capture frame
        t_capture = threading.Thread(target=capture_frames, args=(camera_id,))
        t_capture.daemon = True
        t_capture.start()

        # Thread xử lý YOLO
        t_process = threading.Thread(target=process_frames, args=(camera_id,))
        t_process.daemon = True
        t_process.start()

    # Khởi động Flask server
    print("Starting server at http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, threaded=True)