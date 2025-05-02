import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import cv2

# Load YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # chỉ nhận >=40% độ tin cậy

# Nếu có GPU thì chạy trên GPU
if torch.cuda.is_available():
    model.to('cuda')

# Chỉ lấy các class xe
vehicle_classes = ['car', 'motorbike', 'bus', 'truck']

# Kết nối camera (0 = webcam; hoặc ESP32-CAM IP)
# cap = cv2.VideoCapture('http://192.168.1.61:80/stream')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to connect to camera.")
    exit()

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

    # Hiển thị kết quả
    cv2.imshow('Vehicle Detection', frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
       