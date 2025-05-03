import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import cv2
import threading
from flask import Flask, Response, jsonify, render_template
import time
import queue
import numpy as np

# Flask app setup
app = Flask(__name__)
lock = threading.Lock()

# Cấu hình cameras
CAMERAS = {
    "bai1": {
        "name": "Bãi 1",
        "url": 0,  # 0 = webcam
        "output_frame": None,
        "frame_queue": queue.Queue(maxsize=2),
        "detected_objects": []
    },
    "bai2": {
        "name": "Bãi 2",
        "url": 1,  # 1 = second camera (or another index)
        "output_frame": None,
        "frame_queue": queue.Queue(maxsize=2),
        "detected_objects": []
    }
}

# Cấu hình chung
RESOLUTION = (480, 360)  # Độ phân giải thấp hơn để stream mượt hơn
JPEG_QUALITY = 70  # Giảm chất lượng JPEG để tăng tốc độ truyền
DETECTION_INTERVAL = 2  # Chỉ phát hiện mỗi n frame (giảm xuống 2 để giảm giật)
STREAM_DELAY = 0.03  # Khoảng thời gian giữa các frame (giảm xuống để mượt hơn)

# Load YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # chỉ nhận >=40% độ tin cậy

# Nếu có GPU thì chạy trên GPU
if torch.cuda.is_available():
    model.to('cuda')

# Chỉ lấy các class xe
vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'person']

class DetectedObject:
    def __init__(self, box, label, conf):
        self.box = box  # [x1, y1, x2, y2]
        self.label = label
        self.conf = conf
        self.last_seen = time.time()
        self.frames_tracked = 0
    
    def update(self, box, conf):
        # Làm mượt box bằng cách kết hợp vị trí cũ và mới
        alpha = 0.7  # Hệ số làm mượt (0.5-0.8 là tốt)
        self.box = [
            int(alpha * box[0] + (1 - alpha) * self.box[0]),
            int(alpha * box[1] + (1 - alpha) * self.box[1]),
            int(alpha * box[2] + (1 - alpha) * self.box[2]),
            int(alpha * box[3] + (1 - alpha) * self.box[3])
        ]
        self.conf = conf
        self.last_seen = time.time()
        self.frames_tracked += 1

def detect_vehicles(camera_id):
    camera_config = CAMERAS[camera_id]
    
    # Kết nối camera
    cap = cv2.VideoCapture(camera_config["url"])
    
    # Tăng buffer size cho camera
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print(f"Failed to connect to camera {camera_id}.")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to grab frame from camera {camera_id}.")
            time.sleep(1)  # Wait before retry
            continue

        # Resize ảnh nhỏ lại để tăng hiệu suất
        frame = cv2.resize(frame, RESOLUTION)
        
        # Chỉ chạy YOLOv5 mỗi vài frame để tiết kiệm CPU/GPU
        if frame_count % DETECTION_INTERVAL == 0:
            # YOLO nhận diện
            results = model(frame, size=RESOLUTION[0])
            
            # Lấy kết quả
            new_detections = []
            for *xyxy, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)]
                if label in vehicle_classes:
                    box = list(map(int, xyxy))
                    new_detections.append((box, label, float(conf)))
            
            # Cập nhật danh sách đối tượng đã phát hiện
            update_detected_objects(camera_id, new_detections)
        
        # Vẽ các đối tượng đã phát hiện lên frame
        draw_detections(camera_id, frame)
        
        frame_count += 1
        
        # Cập nhật output_frame có khóa để tránh race condition
        with lock:
            camera_config["output_frame"] = frame.copy()
        
        # Thêm frame vào queue để streaming
        try:
            # Non-blocking, bỏ qua nếu queue đầy để tránh lag
            camera_config["frame_queue"].put(frame.copy(), block=False)
        except queue.Full:
            pass

        # Delay ngắn để giảm tải CPU
        time.sleep(0.01)

    cap.release()

def update_detected_objects(camera_id, new_detections):
    camera_config = CAMERAS[camera_id]
    
    # Xóa các đối tượng quá cũ (không còn xuất hiện trong 1 giây)
    current_time = time.time()
    camera_config["detected_objects"] = [obj for obj in camera_config["detected_objects"] if current_time - obj.last_seen < 1.0]
    
    # Cập nhật hoặc thêm mới các đối tượng
    for box, label, conf in new_detections:
        # Tìm đối tượng phù hợp trong danh sách hiện tại
        matched = False
        for obj in camera_config["detected_objects"]:
            if obj.label == label:
                # Kiểm tra IoU (Intersection over Union) để xác định đó là cùng một đối tượng
                iou = calculate_iou(box, obj.box)
                if iou > 0.3:  # Ngưỡng IoU
                    obj.update(box, conf)
                    matched = True
                    break
        
        # Nếu không tìm thấy đối tượng phù hợp, thêm mới
        if not matched:
            camera_config["detected_objects"].append(DetectedObject(box, label, conf))

def calculate_iou(box1, box2):
    # Tính IoU (Intersection over Union) giữa hai box
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Tính diện tích giao nhau
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # Không có giao nhau
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Tính diện tích hai box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Tính IoU
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def draw_detections(camera_id, frame):
    camera_config = CAMERAS[camera_id]
    
    # Vẽ tất cả các đối tượng đã phát hiện
    for obj in camera_config["detected_objects"]:
        x1, y1, x2, y2 = obj.box
        
        # Vẽ khung
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ nhãn
        cv2.putText(frame, f'{obj.label} {obj.conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def gen_camera_stream(camera_id):
    camera_config = CAMERAS[camera_id]
    prev_frame_time = time.time()
    
    while True:
        # Đợi đến thời điểm thích hợp để gửi frame tiếp theo
        current_time = time.time()
        if current_time - prev_frame_time < STREAM_DELAY:
            time.sleep(0.001)  # Short sleep to reduce CPU usage
            continue
            
        # Lấy frame từ queue hoặc từ output_frame
        frame = None
        try:
            frame = camera_config["frame_queue"].get(block=False)
        except queue.Empty:
            with lock:
                if camera_config["output_frame"] is not None:
                    frame = camera_config["output_frame"].copy()
        
        if frame is None:
            continue
            
        # Mã hóa frame thành JPEG với chất lượng tùy chỉnh
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        (flag, encoded_image) = cv2.imencode(".jpg", frame, encode_param)
        
        if not flag:
            continue
            
        # Cập nhật thời gian frame cuối
        prev_frame_time = time.time()
        
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
