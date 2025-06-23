#!/usr/bin/env python3
"""
Thermal Power Plant Safety Detection System
Main Application Launcher

This script provides a unified interface to run different components of the safety detection system.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import threading
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import cv2
        import ultralytics
        import gradio
        import flask
        print("✅ All dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def run_gradio_app():
    """Run the Gradio web application"""
    print("🚀 Starting Gradio Web Application...")
    print("📱 Open your browser and go to: http://localhost:7860")
    
    os.chdir(Path(__file__).parent / "src")
    subprocess.run([sys.executable, "gradio_app.py"])

def run_flask_api():
    """Run the Flask API server"""
    print("🚀 Starting Flask API Server...")
    print("🔗 API Documentation: http://localhost:5000/api/docs")
    
    os.chdir(Path(__file__).parent / "src")
    subprocess.run([sys.executable, "api.py"])

def run_desktop_app():
    """Run the basic desktop OpenCV application"""
    print("🚀 Starting Basic Desktop Safety Detection System...")
    print("📹 Make sure you have a camera connected")
    print("Press 'q' to quit the application")
    
    os.chdir(Path(__file__).parent / "src")
    subprocess.run([sys.executable, "safety_detector.py"])

def run_enhanced_desktop_app():
    """Run the enhanced desktop application with camera selection"""
    print("🚀 Starting Enhanced Desktop Safety Detection System...")
    print("📹 Features: Camera selection, IP cameras, RTSP, recording, and more")
    print("💡 Use the GUI interface for all controls")
    
    os.chdir(Path(__file__).parent / "src")
    subprocess.run([sys.executable, "enhanced_desktop_app.py"])

def run_training():
    """Run model training"""
    print("🚀 Starting Model Training...")
    print("📚 This will download pre-trained weights and set up training")
    
    os.chdir(Path(__file__).parent / "src")
    subprocess.run([sys.executable, "train_model.py", "--mode", "demo"])

def run_combined_interface():
    """Run both Gradio and Flask API simultaneously"""
    print("🚀 Starting Combined Interface (Gradio + API)...")
    print("📱 Gradio Web App: http://localhost:7860")
    print("🔗 Flask API: http://localhost:5000/api/docs")
    
    # Start Flask API in background thread
    def start_api():
        os.chdir(Path(__file__).parent / "src")
        subprocess.run([sys.executable, "api.py"], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
    
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    # Wait a bit for API to start
    time.sleep(3)
    
    # Start Gradio
    run_gradio_app()

def show_system_info():
    """Display system information"""
    print("\n" + "="*60)
    print("🦺 THERMAL POWER PLANT SAFETY DETECTION SYSTEM")
    print("="*60)
    print("📋 System Components:")
    print("   1. Gradio Web Application - Modern web interface with camera support")
    print("   2. Flask API Server - RESTful API for integration")
    print("   3. Desktop Application - OpenCV-based standalone app")
    print("   4. Model Training - Custom model training pipeline")
    print("   5. Combined Interface - Web app + API together")
    print("\n🎯 Features:")
    print("   • Real-time helmet detection")
    print("   • Reflective jacket/high-vis vest detection") 
    print("   • Safety compliance monitoring")
    print("   • Live video stream processing")
    print("   • Video file analysis")
    print("   • Image analysis")
    print("   • Statistical reporting")
    print("   • Custom model training")
    print("\n🏭 Designed for:")
    print("   • Thermal power plant safety monitoring")
    print("   • Industrial safety compliance")
    print("   • Construction site safety")
    print("   • Real-time safety violations detection")
    print("="*60)


def run_enhanced_detection_test():
    """Test enhanced detection capabilities"""
    print("🧪 Testing Enhanced Safety Detection...")
    print("🎯 Features: 7 PPE classes, environmental adaptation, area-specific rules")
    
    os.chdir(Path(__file__).parent / "src")
    subprocess.run([sys.executable, "enhanced_safety_detector.py"])

def show_enhancement_summary():
    """Show enhancement summary"""
    print("\n" + "="*60)
    print("🏭 THERMAL POWER PLANT ENHANCEMENTS")
    print("="*60)
    print("✅ Critical Issues Addressed:")
    print("   1. Accuracy: Upgraded YOLOv8n → YOLOv8m (+20-30% accuracy)")
    print("   2. PPE Coverage: 3 → 7 classes (added boots, gloves, arc flash, respirator)")
    print("   3. Environmental: Steam, dust, heat, low-light processing")
    print("   4. Area Rules: Switchyard, boiler, coal handling specific requirements")
    print()
    print("🎯 Enhanced Features:")
    print("   • Arc flash suit detection (CRITICAL for electrical areas)")
    print("   • Safety boots detection (required in all areas)")
    print("   • Environmental condition auto-detection")
    print("   • Area-specific PPE requirements")
    print("   • Critical violation alerts")
    print("   • Enhanced accuracy (target: 75%+ mAP)")
    print()
    print("🚀 Ready for:")
    print("   • Thermal power plant deployment")
    print("   • Industrial safety monitoring") 
    print("   • Critical area supervision")
    print("   • Production environments")
    print("="*60)

def main():
    """Main application launcher"""
    
    parser = argparse.ArgumentParser(
        description="Thermal Power Plant Safety Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode web          # Start web interface
  python run.py --mode api          # Start API server
  python run.py --mode desktop      # Start desktop app
  python run.py --mode train        # Start training
  python run.py --mode combined     # Start web + API
  python run.py --info              # Show system info
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['web', 'api', 'desktop', 'enhanced-desktop', 'train', 'combined', 'enhanced-test', 'show-enhancements'],
        default='web',
        help='Application mode to run (default: web)'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show system information and exit'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    args = parser.parse_args()
    
    # Show system info
    if args.info:
        show_system_info()
        return
    
    # Check dependencies
    if args.check_deps:
        check_dependencies()
        return
    
    # Check dependencies before running
    if not check_dependencies():
        print("\n💡 Install dependencies with:")
        print("   pip install -r requirements.txt")
        return
    
    # Show system info by default
    show_system_info()
    
    print(f"\n🚀 Starting in {args.mode.upper()} mode...\n")
    
    # Run selected mode
    try:
        if args.mode == 'web':
            run_gradio_app()
        
        elif args.mode == 'api':
            run_flask_api()
        
        elif args.mode == 'desktop':
            run_desktop_app()
        
        elif args.mode == 'enhanced-desktop':
            run_enhanced_desktop_app()
        
        elif args.mode == 'train':
            run_training()
        
        elif args.mode == 'combined':
            run_combined_interface()
        
        elif args.mode == 'enhanced-test':
            run_enhanced_detection_test()
        
        elif args.mode == 'show-enhancements':
            show_enhancement_summary()
            
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        print("💡 Try checking the logs for more details")

if __name__ == "__main__":
    main()
