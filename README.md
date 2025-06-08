# Object Detection Project

Dự án phát hiện đối tượng sử dụng YOLOv8 và PyTorch.

## Cài đặt

1. Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
.\venv\Scripts\activate  # Windows
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

3. Tải model YOLOv8:
```bash
# Model sẽ được tự động tải khi chạy lần đầu
# Hoặc bạn có thể tải thủ công:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## Sử dụng

1. Phát hiện đối tượng trong ảnh:
```bash
python src/detect.py --source path/to/image.jpg
```

2. Phát hiện đối tượng trong video:
```bash
python src/detect.py --source path/to/video.mp4
```

3. Phát hiện đối tượng từ webcam:
```bash
python src/detect.py --source 0
```

## Tính năng

- Phát hiện đối tượng trong ảnh tĩnh
- Phát hiện đối tượng trong video
- Hỗ trợ webcam
- Hiển thị kết quả với bounding boxes và nhãn
- Lưu kết quả ra file

## Model

Dự án sử dụng YOLOv8, một trong những model object detection mạnh nhất hiện nay:
- Tốc độ xử lý nhanh
- Độ chính xác cao
- Hỗ trợ nhiều loại đối tượng
- Dễ dàng fine-tune cho các use case cụ thể

## License

Dự án này được cấp phép theo MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết. 