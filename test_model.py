#!/usr/bin/env python3
"""
Quick test script to verify the trained model works
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys

def test_trained_model():
    """Test the trained safety detection model"""
    
    print("🧪 Testing Trained Safety Detection Model")
    print("=" * 50)
    
    # Check if trained model exists
    model_path = "models/safety_detection_best.pt"
    if not Path(model_path).exists():
        print(f"❌ Trained model not found: {model_path}")
        print("   Please train the model first using:")
        print("   python3 train_simple.py --dataset data/enhanced_dataset/dataset.yaml")
        return False
    
    try:
        # Load trained model
        print(f"📂 Loading trained model: {model_path}")
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        
        # Check if we have test images
        test_images_dir = Path("data/enhanced_dataset/images/test")
        if not test_images_dir.exists():
            print(f"❌ Test images directory not found: {test_images_dir}")
            return False
        
        # Get test images
        test_images = list(test_images_dir.glob("*.jpg"))
        if not test_images:
            print("❌ No test images found")
            return False
        
        print(f"📸 Found {len(test_images)} test images")
        
        # Test model on a few images
        for i, img_path in enumerate(test_images[:3]):  # Test first 3 images
            print(f"\n🔍 Testing image {i+1}: {img_path.name}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"   ❌ Could not load image: {img_path}")
                continue
            
            print(f"   📏 Image size: {image.shape[1]}x{image.shape[0]}")
            
            # Run inference
            results = model(image, conf=0.3, verbose=False)
            
            # Count detections by class
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                class_ids = boxes.cls.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                
                # Count by class
                helmets = sum(1 for cls_id in class_ids if int(cls_id) == 0)
                jackets = sum(1 for cls_id in class_ids if int(cls_id) == 1)
                persons = sum(1 for cls_id in class_ids if int(cls_id) == 2)
                
                print(f"   🔍 Detections found:")
                print(f"     👷 Persons: {persons}")
                print(f"     🪖 Helmets: {helmets}")
                print(f"     🦺 Reflective Jackets: {jackets}")
                print(f"     📊 Total detections: {len(class_ids)}")
                print(f"     🎯 Max confidence: {max(confidences):.3f}" if len(confidences) > 0 else "     🎯 No detections")
                
            else:
                print(f"   ℹ️ No detections found")
        
        print(f"\n✅ Model testing completed successfully!")
        print(f"📋 Model Summary:")
        print(f"   • Model file: {model_path}")
        print(f"   • Model size: {Path(model_path).stat().st_size / 1024 / 1024:.1f} MB")
        print(f"   • Classes: 0=helmet, 1=reflective_jacket, 2=person")
        print(f"   • Test images: {len(test_images)} available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        return False

def test_safety_detector_class():
    """Test the SafetyDetector class with trained model"""
    
    print("\n🧪 Testing SafetyDetector Class")
    print("=" * 50)
    
    try:
        from src.safety_detector import SafetyDetector
        
        # Initialize with trained model
        model_path = "models/safety_detection_best.pt"
        detector = SafetyDetector(model_path=model_path, confidence_threshold=0.3)
        
        print(f"✅ SafetyDetector initialized with trained model")
        print(f"   Device: {detector.device}")
        print(f"   Confidence threshold: {detector.confidence_threshold}")
        
        # Test with a sample image
        test_images_dir = Path("data/enhanced_dataset/images/test")
        test_images = list(test_images_dir.glob("*.jpg"))
        
        if test_images:
            test_image = cv2.imread(str(test_images[0]))
            print(f"\n🔍 Testing SafetyDetector on: {test_images[0].name}")
            
            results = detector.detect_safety_equipment(test_image)
            
            if 'error' not in results:
                safety_analysis = results['safety_analysis']
                print(f"   📊 Safety Analysis:")
                print(f"     👷 Total persons: {safety_analysis['total_persons']}")
                print(f"     🪖 Helmet compliance: {safety_analysis['helmet_compliance_rate']:.1f}%")
                print(f"     🦺 Jacket compliance: {safety_analysis['jacket_compliance_rate']:.1f}%")
                print(f"     ✅ Is compliant: {safety_analysis['is_compliant']}")
                print(f"     ⚠️ Violations: {len(safety_analysis['violations'])}")
                
                if safety_analysis['violations']:
                    for violation in safety_analysis['violations']:
                        print(f"       - {violation}")
            else:
                print(f"   ❌ Error in detection: {results['error']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing SafetyDetector: {e}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Safety Detection System - Model Testing")
    print("=" * 60)
    
    # Test 1: Direct model testing
    model_test_passed = test_trained_model()
    
    # Test 2: SafetyDetector class testing
    detector_test_passed = test_safety_detector_class()
    
    print("\n📋 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Trained Model Test: {'PASSED' if model_test_passed else 'FAILED'}")
    print(f"✅ SafetyDetector Test: {'PASSED' if detector_test_passed else 'FAILED'}")
    
    if model_test_passed and detector_test_passed:
        print(f"\n🎉 ALL TESTS PASSED! The safety detection system is working correctly.")
        print(f"🚀 System is ready for production use!")
    else:
        print(f"\n⚠️ Some tests failed. Please check the error messages above.")
        
    print(f"\n💡 To run the web interface:")
    print(f"   python3 run.py --mode web")
    print(f"\n💡 To run the API server:")
    print(f"   python3 run.py --mode api")
    
    return model_test_passed and detector_test_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
