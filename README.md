# AI Object Detection

Real-time object detection system using YOLOv8 with support for images, videos, and webcam feeds.

## Features

- 🚀 Real-time object detection
- 📷 Multiple input sources support:
  - Image files (jpg, png, etc.)
  - Video files (mp4, avi, etc.)
  - Webcam feed
  - RTSP streams
  - Video URLs
- 🎯 Adjustable confidence threshold
- 💾 Save detection results
- 🎨 Visual output with bounding boxes and labels
- 🔍 Supports 80 COCO dataset object classes
- 🔄 Object tracking support
- 🎯 Class filtering
- 📊 JSON export
- 🎨 Customizable bounding box colors

## Requirements

- Python 3.10.14
- OpenCV
- Ultralytics YOLOv8
- NumPy
- CUDA (recommended for GPU)
- Webcam or video source

## Installation

1. Clone the repository:
```bash
git clone https://github.com/wsxyanua/AI-OBJ.git
cd AI-OBJ
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install model YOLOv8:
```bash
# Install model YOLOv8n (lightest version)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Or download model YOLOv3
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

## Usage

### Basic Usage

1. Image detection:
```bash
python src/detect.py --source path/to/image.jpg
```

2. Video detection:
```bash
python src/detect.py --source path/to/video.mp4
```

3. Webcam detection:
```bash
python src/detect.py --source 0
```

4. RTSP stream detection:
```bash
python src/detect.py --source rtsp://username:password@ip:port/stream
```

### Advanced Usage

1. Object Tracking:
```bash
python3 src/detect.py --source 0 --track
```

2. Class Filtering:
```bash
# Detect only persons, bicycles, and cars
python3 src/detect.py --source video.mp4 --classes person,bicycle,car
```

3. JSON Export:
```bash
python3 src/detect.py --source video.mp4 --export-json
```

4. Custom Colors:
```bash
python3 src/detect.py --source 0 --colors "255,0,0;0,255,0;0,0,255"
```

5. Adjust Confidence:
```bash
python3 src/detect.py --source 0 --conf 0.5
```

6. Save Results:
```bash
python3 src/detect.py --source video.mp4 --save --output results
```

## Detectable Objects

Supports 80 object classes including:

- People and animals
- Vehicles
- Electronics
- Household items
- Sports equipment
- Food items

Full list available in 'coco.names'.

## Model

Uses YOLOv8n (nano) model, offering:
- Fast inference speed
- Good accuracy
- Low resource requirements
- Pre-trained on COCO dataset

## Controls

- Press 'q' to quit the detection window
- Detection results are displayed in real-time

## Performance

- Preprocessing time: ~1-3ms
- Inference time: ~70-90ms
- Postprocessing time: ~0.8-2ms
- Total processing time per frame: ~80-100ms
- Resolution: 480x640 pixels

## Notes

- Ensure webcam or video source is available and working
- Check webcam permissions
- Adjust confidence threshold based on needs
- Use GPU for better performance
- For Linux users, you can install dependencies using apt:
```bash
sudo apt install python3-opencv python3-numpy python3-pip python3-torch python3-torchvision python3-pillow
pip3 install ultralytics --break-system-packages
```

## License

This project is licensed under the MIT License - see file [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request
