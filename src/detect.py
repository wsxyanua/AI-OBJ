import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection using YOLOv8')
    parser.add_argument('--source', type=str, required=True,
                      help='Path to image/video file or webcam index (0)')
    parser.add_argument('--conf', type=float, default=0.25,
                      help='Confidence threshold')
    parser.add_argument('--save', action='store_true',
                      help='Save results')
    parser.add_argument('--output', type=str, default='output',
                      help='Output directory')
    return parser.parse_args()

def process_frame(frame, model, conf_threshold):
    # Perform detection
    results = model(frame, conf=conf_threshold)[0]
    
    # Draw results
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Get class name
        class_name = results.names[int(class_id)]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f'{class_name}: {score:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.save:
        os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open video capture
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open source {args.source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer if saving
    writer = None
    if args.save and not args.source.isdigit():
        output_path = os.path.join(args.output, f"output_{os.path.basename(args.source)}")
        try:
            writer = cv2.VideoWriter(output_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (width, height))
            if not writer.isOpened():
                raise Exception("Failed to create video writer")
        except Exception as e:
            print(f"Error creating video writer: {e}")
            return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = process_frame(frame, model, args.conf)
            
            # Show result
            cv2.imshow('Object Detection', processed_frame)
            
            # Save if needed
            if args.save:
                if args.source.isdigit():
                    output_path = os.path.join(args.output, f"webcam_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg")
                    cv2.imwrite(output_path, processed_frame)
                elif writer is not None:
                    writer.write(processed_frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 