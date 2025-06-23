#!/usr/bin/env python3
"""
Extract frames from videos for YOLO training dataset
"""

import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, frame_interval=30):
    """
    Extract frames from video at specified intervals
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (default: 30)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Extracting every {frame_interval}th frame")
    
    frame_count = 0
    saved_count = 0
    
    # Get video filename without extension for naming frames
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0:
            # Create filename with video name and frame number
            frame_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Extracted {saved_count} frames...")
        
        frame_count += 1
    
    cap.release()
    print(f"Extraction complete! Saved {saved_count} frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos for YOLO training')
    parser.add_argument('--video-dir', default='/Users/soumyajitghosh/safety_detection_system',
                       help='Directory containing video files')
    parser.add_argument('--downloads-dir', default='/Users/soumyajitghosh/Downloads',
                       help='Downloads directory containing additional video files')
    parser.add_argument('--output-dir', default='/Users/soumyajitghosh/safety_detection_system/data/images/train',
                       help='Output directory for extracted frames')
    parser.add_argument('--interval', type=int, default=30,
                       help='Extract every Nth frame (default: 30)')
    
    args = parser.parse_args()
    
    # Video files to process from project directory
    project_video_files = [
        'safety_detection_fixed_output.mp4',
        'src/safety_detection_output.mp4'
    ]
    
    # Additional video files from Downloads directory
    downloads_video_files = [
        'NVR_ch50_main_20250621080018_20250621081559.mp4',
        'NVR_ch53_main_20250621074500_20250621080019.mp4',
        'NVR_ch56_main_20250621075900_20250621080021.mp4'
    ]
    
    print("Processing videos from project directory...")
    for video_file in project_video_files:
        video_path = os.path.join(args.video_dir, video_file)
        
        if os.path.exists(video_path):
            extract_frames(video_path, args.output_dir, args.interval)
        else:
            print(f"Warning: Video file not found: {video_path}")
    
    print("\nProcessing videos from Downloads directory...")
    for video_file in downloads_video_files:
        video_path = os.path.join(args.downloads_dir, video_file)
        
        if os.path.exists(video_path):
            extract_frames(video_path, args.output_dir, args.interval)
        else:
            print(f"Warning: Video file not found: {video_path}")

if __name__ == "__main__":
    main()
