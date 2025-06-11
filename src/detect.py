import cv2
import numpy as np
import argparse
import os
import json
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

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
    parser.add_argument('--track', action='store_true',
                      help='Enable object tracking')
    parser.add_argument('--classes', type=str, default=None,
                      help='Filter by class names (comma-separated)')
    parser.add_argument('--export-json', action='store_true',
                      help='Export detection results to JSON')
    parser.add_argument('--colors', type=str, default=None,
                      help='Custom colors for classes (comma-separated RGB values)')
    return parser.parse_args()

def get_colors(classes, custom_colors=None):
    if custom_colors:
        color_list = [tuple(map(int, color.split(','))) for color in custom_colors.split(';')]
        return {cls: color_list[i % len(color_list)] for i, cls in enumerate(classes)}
    return {cls: tuple(np.random.randint(0, 255, 3).tolist()) for cls in classes}

def process_frame(frame, model, conf_threshold, tracker=None, class_filter=None, colors=None):
    # Perform detection
    results = model(frame, conf=conf_threshold)[0]
    
    # Initialize tracking if enabled
    if tracker is not None:
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if class_filter is None or results.names[int(class_id)] in class_filter:
                detections.append([x1, y1, x2, y2, score])
        
        if len(detections) > 0:
            tracks = tracker.update(np.array(detections), frame)
        else:
            tracks = []
    else:
        tracks = results.boxes.data.tolist()

    # Process results
    detections_info = []
    for track in tracks:
        if tracker is not None:
            x1, y1, x2, y2, track_id = track
            score = 1.0  # Tracking doesn't provide confidence score
        else:
            x1, y1, x2, y2, score, class_id = track
            track_id = None
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Get class name
        class_name = results.names[int(class_id)] if tracker is None else class_filter[0]
        
        # Skip if class is filtered
        if class_filter is not None and class_name not in class_filter:
            continue
        
        # Get color for this class
        color = colors.get(class_name, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label_parts = [class_name]
        if score is not None:
            label_parts.append(f'{score:.2f}')
        if track_id is not None:
            label_parts.append(f'ID: {int(track_id)}')
        label = ' '.join(label_parts)
        
        # Draw label with background
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Store detection info
        detections_info.append({
            'class': class_name,
            'confidence': float(score) if score is not None else None,
            'track_id': int(track_id) if track_id is not None else None,
            'bbox': [x1, y1, x2, y2]
        })
    
    return frame, detections_info

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if args.save or args.export_json:
        os.makedirs(args.output, exist_ok=True)
    
    try:
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize tracker if enabled
    tracker = None
    if args.track:
        tracker = cv2.TrackerCSRT_create()
    
    # Parse class filter
    class_filter = None
    if args.classes:
        class_filter = [cls.strip() for cls in args.classes.split(',')]
    
    # Get colors for classes
    colors = get_colors(model.names.values(), args.colors)
    
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
    
    # Initialize JSON export
    json_data = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            processed_frame, detections_info = process_frame(
                frame, model, args.conf, tracker, class_filter, colors
            )
            
            # Store detection info for JSON export
            if args.export_json:
                json_data.append({
                    'frame': frame_count,
                    'timestamp': datetime.now().isoformat(),
                    'detections': detections_info
                })
            
            # Show result
            cv2.imshow('Object Detection', processed_frame)
            
            # Save if needed
            if args.save:
                if args.source.isdigit():
                    output_path = os.path.join(args.output, f"webcam_{frame_count}.jpg")
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
        
        # Export JSON if requested
        if args.export_json:
            json_path = os.path.join(args.output, f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)

if __name__ == '__main__':
    main() 