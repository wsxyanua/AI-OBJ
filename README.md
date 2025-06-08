# AI Object Detection

Real-time object detection system using YOLOv8 with support for images, videos, and webcam feeds.

## Features

- 🚀 Real-time object detection
- 📷 Multiple input sources support:
  - Image files (jpg, png, etc.)
  - Video files (mp4, avi, etc.)
  - Webcam feed
- 🎯 Adjustable confidence threshold
- 💾 Save detection results
- 🎨 Visual output with bounding boxes and labels
- 🔍 Supports 80 COCO dataset object classes

## Requirements

- Python 3.10.14
- OpenCV
- Ultralytics YOLOv8
- NumPy

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
Basic Usage
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

Advanced Usage:
1. Set confidence threshold:
```bash
python src/detect.py --source video.mp4 --conf 0.25
```

2. Save detection results:
```bash
python src/detect.py --source image.jpg --save
```

3. Specify output directory:
```bash
python src/detect.py --source video.mp4 --save --output results
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
- Detection relsult are display in real-time

## License

This project is license under the MIT License - see file [LICENSE](LICENSE) file for detail

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request
