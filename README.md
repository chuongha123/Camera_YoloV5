# Camera YOLOv5 Vehicle Detection

Hệ thống phát hiện xe (car, motorbike, bus, truck) và người (person) qua nhiều webcam hoặc IP camera sử dụng YOLOv5 và phát trực tiếp qua HTTP cho ứng dụng di động.

## Yêu cầu hệ thống

- Python 3.8+
- Một hoặc nhiều webcam hoặc IP camera
- GPU (tùy chọn, nhưng khuyến khích để tăng hiệu suất)

## Cài đặt

1. Clone repository hoặc tải về:
   ```
   git clone <repository-url>
   cd Camera_YoloV5
   ```

2. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```

## Chạy ứng dụng

1. Khởi động server phát hiện và stream:
   ```
   python main.py
   ```

2. Server sẽ khởi động tại địa chỉ http://0.0.0.0:5000
   - Giao diện web multi-camera: http://localhost:5000
   - API thông tin: http://localhost:5000/api/info
   - Stream phát hiện trực tiếp: http://localhost:5000/video_feed/bai1

## Cấu hình nhiều camera

Mặc định, hệ thống được cấu hình để sử dụng 2 camera, có thể thêm hoặc sửa trong file `main.py`:

```python
# Cấu hình cameras
CAMERAS = {
    "bai1": {
        "name": "Bãi 1",
        "url": 0,  # 0 = webcam đầu tiên
        "output_frame": None,
        "frame_queue": queue.Queue(maxsize=2),
        "detected_objects": []
    },
    "bai2": {
        "name": "Bãi 2",
        "url": 1,  # 1 = webcam thứ hai
        "output_frame": None,
        "frame_queue": queue.Queue(maxsize=2),
        "detected_objects": []
    }
}
```

Để thêm camera mới:
- Thêm một cặp key-value vào từ điển `CAMERAS`
- `url` có thể là số index (0, 1, 2) hoặc URL của IP camera

Ví dụ thêm camera thứ 3:
```python
"bai3": {
    "name": "Bãi 3",
    "url": "http://192.168.1.100:8080/video",
    "output_frame": None,
    "frame_queue": queue.Queue(maxsize=2),
    "detected_objects": []
}
```

## Kết nối từ thiết bị di động

### Giao diện web
Truy cập địa chỉ IP của máy tính từ trình duyệt điện thoại:
```
http://YOUR_PC_IP:5000
```
Giao diện web sẽ hiển thị tất cả các camera được cấu hình, mỗi camera có nhãn riêng.

### Cấu hình camera
- Đổi camera (trong file main.py):
  ```python
  # Webcam (mặc định)
  "url": 0
  
  # Hoặc IP Camera
  "url": "http://camera-ip-address:port/stream"
  ```

### Kết nối từ React Native

1. Cài đặt React Native WebView:
   ```
   npm install react-native-webview
   ```

2. Tạo component hiển thị stream:
   ```javascript
   import React from 'react';
   import { View, StyleSheet } from 'react-native';
   import { WebView } from 'react-native-webview';

   export default function CameraStream() {
     // Thay YOUR_PC_IP bằng địa chỉ IP máy tính chạy Python
     const streamUrl = 'http://YOUR_PC_IP:5000/video_feed/bai1';
     
     return (
       <View style={styles.container}>
         <WebView 
           source={{ uri: streamUrl }}
           style={styles.webview}
         />
       </View>
     );
   }

   const styles = StyleSheet.create({
     container: {
       flex: 1,
     },
     webview: {
       flex: 1,
     },
   });
   ```

   Hoặc sử dụng giao diện web có sẵn để xem nhiều camera:
   ```javascript
   const webUrl = 'http://YOUR_PC_IP:5000';
   
   return (
     <View style={styles.container}>
       <WebView 
         source={{ uri: webUrl }}
         style={styles.webview}
       />
     </View>
   );
   ```

## Xử lý sự cố

1. **Lỗi kết nối camera**:
   - Kiểm tra camera đã được kết nối đúng chưa
   - Thử chuyển sang index khác: `"url": 1` hoặc `"url": 2`
   - Nếu dùng nhiều camera, cần đảm bảo số lượng camera vật lý phải bằng hoặc lớn hơn số camera cấu hình

2. **Không thể kết nối từ thiết bị di động**:
   - Đảm bảo thiết bị di động và máy tính đang ở cùng một mạng
   - Kiểm tra tường lửa có chặn cổng 5000 không
   - Sử dụng địa chỉ IP thực tế của máy chạy server (dùng lệnh `ipconfig` trên Windows)

3. **Hiệu suất thấp**:
   - Giảm kích thước frame trong main.py: `RESOLUTION = (320, 240)`
   - Tăng `DETECTION_INTERVAL` lên cao hơn để giảm tần suất phát hiện
   - Tăng `STREAM_DELAY` để giảm tần suất frame
   - Nếu có GPU, đảm bảo model được chạy trên GPU