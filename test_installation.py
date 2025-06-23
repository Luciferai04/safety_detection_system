#!/usr/bin/env python3
"""
Installation Test Script for Safety Detection System

This script verifies that all components are properly installed and functional.
"""

import sys
import os
import importlib
from pathlib import Path
import subprocess

def test_python_version():
    """Test Python version compatibility"""
    print("🐍 Testing Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'ultralytics',
        'cv2',
        'numpy',
        'PIL',
        'flask',
        'gradio',
        'pandas',
        'matplotlib',
        'plotly',
        'yaml'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'yaml':
                import yaml
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package} - {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n💡 Install missing packages: pip install {' '.join(failed_imports)}")
        return False
    
    return True

def test_torch_device():
    """Test PyTorch device availability"""
    print("\n🖥️ Testing PyTorch devices...")
    
    try:
        import torch
        
        # Test CPU
        print(f"✅ CPU available")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️ CUDA not available")
        
        # Test MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("ℹ️ MPS not available")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch device test failed: {e}")
        return False

def test_opencv_camera():
    """Test OpenCV camera access"""
    print("\n📹 Testing camera access...")
    
    try:
        import cv2
        
        # Try to open default camera
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera accessible and functional")
                print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("⚠️ Camera opened but no frame captured")
            cap.release()
            return True
        else:
            print("ℹ️ No camera detected (optional for file-based processing)")
            return True
            
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_yolo_model():
    """Test YOLO model loading"""
    print("\n🤖 Testing YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        
        # Try to load a small YOLO model
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 nano model loaded successfully")
        
        # Test inference on dummy data
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        print("✅ Model inference test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO model test failed: {e}")
        return False

def test_safety_detector():
    """Test our safety detector class"""
    print("\n🦺 Testing SafetyDetector class...")
    
    try:
        # Add src directory to path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        from safety_detector import SafetyDetector
        
        # Initialize detector
        detector = SafetyDetector(confidence_threshold=0.5)
        print("✅ SafetyDetector initialized successfully")
        
        # Test with dummy image
        import numpy as np
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        results = detector.detect_safety_equipment(dummy_image)
        print("✅ Safety detection test successful")
        print(f"   Detection results: {len(results.get('detections', []))} objects")
        
        return True
        
    except Exception as e:
        print(f"❌ SafetyDetector test failed: {e}")
        return False

def test_web_components():
    """Test web components availability"""
    print("\n🌐 Testing web components...")
    
    try:
        # Test Gradio
        import gradio as gr
        print("✅ Gradio available")
        
        # Test Flask
        from flask import Flask
        print("✅ Flask available")
        
        # Test plotting libraries
        import plotly.express as px
        print("✅ Plotly available")
        
        return True
        
    except Exception as e:
        print(f"❌ Web components test failed: {e}")
        return False

def test_configuration():
    """Test configuration files"""
    print("\n⚙️ Testing configuration...")
    
    try:
        config_path = Path(__file__).parent / "config" / "config.yaml"
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            print("✅ Configuration file loaded successfully")
            print(f"   Model: {config.get('model', {}).get('name', 'Unknown')}")
            return True
        else:
            print("⚠️ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_directory_structure():
    """Test project directory structure"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "src",
        "config", 
        "models",
        "data",
        "logs"
    ]
    
    base_path = Path(__file__).parent
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n💡 Create missing directories: mkdir -p {' '.join(missing_dirs)}")
        return False
    
    return True

def run_all_tests():
    """Run all tests and report results"""
    print("="*60)
    print("🦺 SAFETY DETECTION SYSTEM - INSTALLATION TEST")
    print("="*60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("PyTorch Devices", test_torch_device),
        ("Camera Access", test_opencv_camera),
        ("YOLO Model", test_yolo_model),
        ("Safety Detector", test_safety_detector),
        ("Web Components", test_web_components),
        ("Configuration", test_configuration),
        ("Directory Structure", test_directory_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} - Unexpected error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\n🚀 Quick start:")
        print("   python run.py --mode web")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix issues before using the system.")
        print("\n💡 Common fixes:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Create missing directories: mkdir -p models data logs")
        return False

def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("Safety Detection System Installation Test")
        print("\nUsage: python test_installation.py")
        print("\nThis script tests all system components to ensure proper installation.")
        return
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
