# Camera YOLOv5 Vehicle Detection

Hệ thống phát hiện xe (car, motorbike, bus, truck) và người (person) qua webcam hoặc IP camera sử dụng YOLOv5 và phát trực tiếp qua HTTP cho ứng dụng di động.

## Yêu cầu hệ thống

- Python 3.8+
- Webcam hoặc IP camera
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
   - Giao diện web fullscreen: http://localhost:5000
   - API thông tin: http://localhost:5000/api/info
   - Stream phát hiện trực tiếp: http://localhost:5000/video_feed

## Kết nối từ thiết bị di động

### Giao diện web
Truy cập địa chỉ IP của máy tính từ trình duyệt điện thoại:
```
http://YOUR_PC_IP:5000
```
Giao diện web đã được tối ưu cho thiết bị di động, camera sẽ hiển thị toàn màn hình.

### Cấu hình camera
- Đổi camera (trong file main.py):
  ```python
  # Webcam (mặc định)
  cap = cv2.VideoCapture(0)
  
  # Hoặc IP Camera
  # cap = cv2.VideoCapture('http://camera-ip-address:port/stream')
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
     const streamUrl = 'http://YOUR_PC_IP:5000/video_feed';
     
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

   Hoặc sử dụng giao diện web có sẵn:
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
   - Thử chuyển sang index khác: `cv2.VideoCapture(1)` hoặc `cv2.VideoCapture(2)`

2. **Không thể kết nối từ thiết bị di động**:
   - Đảm bảo thiết bị di động và máy tính đang ở cùng một mạng
   - Kiểm tra tường lửa có chặn cổng 5000 không
   - Sử dụng địa chỉ IP thực tế của máy chạy server (dùng lệnh `ipconfig` trên Windows)

3. **Hiệu suất thấp**:
   - Giảm kích thước frame trong main.py: `frame = cv2.resize(frame, (320, 240))`
   - Tăng thời gian trễ: `time.sleep(0.05)`
   - Nếu có GPU, đảm bảo model được chạy trên GPU