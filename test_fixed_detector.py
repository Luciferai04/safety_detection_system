#!/usr/bin/env python3
"""
Test the Fixed Safety Detection System
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from safety_detector_fixed import SafetyDetectorFixed

def main():
    print("🦺 TESTING FIXED SAFETY DETECTION SYSTEM")
    print("=" * 60)
    print("🔧 This version uses:")
    print("   • YOLO for person detection")
    print("   • Color-based helmet detection (yellow, orange, white, red)")
    print("   • Color-based high-vis jacket detection (yellow, orange, lime)")
    print("   • More lenient compliance thresholds")
    print("=" * 60)
    print()
    print("📹 Starting camera feed...")
    print("💡 To test:")
    print("   • Wear bright yellow/orange colors for detection")
    print("   • Press 'q' to quit")
    print()
    
    try:
        # Initialize fixed safety detector with lower confidence for better detection
        detector = SafetyDetectorFixed(confidence_threshold=0.3)
        
        # Test with webcam
        detector.process_video_stream(
            video_source=0,
            save_output=True,
            output_path="safety_detection_fixed_output.mp4"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you have a camera connected")
        print("💡 Try adjusting lighting conditions for better color detection")

if __name__ == "__main__":
    main()
