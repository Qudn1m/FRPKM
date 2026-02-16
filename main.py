import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
import os
import io
import random
from io import BytesIO
from PIL import Image
from base64 import b64decode, b64encode



def start_webcam_local():
    """Capture video from local webcam"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return None
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print("Recording... Press 'q' to stop")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        print(f"Captured {len(frames)} frames")
        
        # Display frame
        cv2.imshow('Webcam', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Total frames captured: {len(frames)}")  # Debug output
    return frames, fps



def save_video(frames, fps, output_path='temp_video.mp4'):
    """Save frames to video file"""
    if not frames:
        print("No frames to save")
        return
    
    frame_height, frame_width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

cascade_path = r'D:\Code\FRPKM\opencv\data\haarcascades\haarcascade_frontalface_default.xml'

def face_tracking(video_path, cascade_path=cascade_path, fps=30):
    """Track faces in video and return detection data"""
    
    if not os.path.exists(cascade_path):
        print(f"Error: Cascade file not found at {cascade_path}")
        return None
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    frame_count = 0
    face_counts = []  # Track number of faces per frame
    frame_times = []  # Track time in seconds
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        
        # Track detection data
        num_faces = len(faces)
        face_counts.append(num_faces)
        frame_times.append(frame_count / fps)  # Convert frame number to time in seconds
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display every 30th frame
        if frame_count % 30 == 0:
            cv2.imshow('Face Detection', frame)
        
        frame_count += 1
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")
    
    return face_counts, frame_times

def plot_face_detection(face_counts, frame_times):
    """Create visualization of face detection over time"""
    plt.figure(figsize=(12, 6))
    
    # Line plot
    plt.plot(frame_times, face_counts, linewidth=2, color='blue', label='Number of Faces')
    plt.fill_between(frame_times, face_counts, alpha=0.3, color='blue')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Number of Faces Detected', fontsize=12)
    plt.title('Face Detection Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('face_detection_diagram.png', dpi=300, bbox_inches='tight')
    print("Diagram saved as 'face_detection_diagram.png'")
    plt.show()
        
# Main execution
if __name__ == "__main__":
    # Capture webcam
    print("Starting webcam capture...")
    result = start_webcam_local()
    
    if result:
        frames, fps = result
        
        # Save video
        random.seed(1)
        output_path = f"temp_video_{random.randint(1000, 9999)}.mp4"
        save_video(frames, fps, output_path)
        
        # Process with face detection
        if os.path.exists(output_path):
            print("Starting face detection...")
            detection_data = face_tracking(output_path, cascade_path, fps)
            
            # Create diagram if detection data is available
            if detection_data:
                face_counts, frame_times = detection_data
                print("Creating face detection diagram...")
                plot_face_detection(face_counts, frame_times)