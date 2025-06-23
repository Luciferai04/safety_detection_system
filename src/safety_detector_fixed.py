import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import time
import math

class SafetyDetectorFixed:
    """
    Fixed Safety Detection System that works with pre-trained YOLO models
    
    This version implements a pragmatic approach using available COCO classes
    and color-based detection for safety equipment until a custom model is trained.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.3,  # Lower threshold for better detection
                 iou_threshold: float = 0.45,
                 device: str = 'auto'):
        """
        Initialize the Fixed Safety Detection System
        """
        # Validate input parameters
        if not 0 < confidence_threshold < 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 < iou_threshold < 1:
            raise ValueError("iou_threshold must be between 0 and 1")
            
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = self._setup_device(device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load YOLO model
        self.model = self._load_model(model_path)
        
        # Define color ranges for safety equipment detection (HSV)
        self.safety_colors = {
            'helmet': {
                'yellow': [(20, 100, 100), (30, 255, 255)],
                'orange': [(10, 100, 100), (20, 255, 255)],
                'white': [(0, 0, 200), (180, 30, 255)],
                'red': [(0, 100, 100), (10, 255, 255)]
            },
            'high_vis': {
                'yellow': [(20, 100, 100), (30, 255, 255)],
                'orange': [(10, 100, 100), (20, 255, 255)],
                'lime': [(35, 100, 100), (80, 255, 255)]
            }
        }
        
        # Safety status tracking
        self.violation_count = 0
        self.total_detections = 0
        
        # Performance monitoring
        self.processing_times = []
        self.max_processing_times = 1000
        
        self.logger.info("Fixed Safety Detector initialized - using color-based detection")
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # Apple Silicon
            else:
                return 'cpu'
        return device
    
    def _load_model(self, model_path: Optional[str]) -> YOLO:
        """Load YOLO model for safety detection"""
        try:
            if model_path and Path(model_path).exists():
                model = YOLO(model_path)
                self.logger.info(f"Loaded custom model from {model_path}")
            else:
                # Use pre-trained YOLOv8 model
                model = YOLO('yolov8n.pt')
                self.logger.info("Loaded pre-trained YOLOv8 model (using color-based safety detection)")
            
            # Move model to appropriate device
            model.to(self.device)
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def detect_safety_equipment(self, frame: np.ndarray) -> Dict:
        """
        Detect safety equipment using YOLO + color-based approach
        """
        start_time = time.time()
        
        try:
            # Input validation
            if frame is None or frame.size == 0:
                raise ValueError("Invalid input frame")
            
            # Run YOLO inference to detect persons
            results = self.model(frame, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               verbose=False)
            
            # Process YOLO detections
            yolo_detections = self._process_yolo_detections(results[0], frame.shape)
            
            # Add color-based safety equipment detection
            safety_detections = self._detect_safety_equipment_by_color(frame, yolo_detections)
            
            # Combine all detections
            all_detections = yolo_detections + safety_detections
            
            # Analyze safety compliance
            safety_analysis = self._analyze_safety_compliance(all_detections, frame)
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_times:
                self.processing_times.pop(0)
            
            # Update statistics
            self.total_detections += 1
            if not safety_analysis.get('is_compliant', True):
                self.violation_count += 1
            
            return {
                'detections': all_detections,
                'safety_analysis': safety_analysis,
                'frame_shape': frame.shape,
                'timestamp': time.time(),
                'processing_time': processing_time,
                'device': self.device
            }
            
        except Exception as e:
            self.logger.error(f"Detection error: {e}")
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': time.time(),
                'processing_time': time.time() - start_time
            }

    def _process_yolo_detections(self, result, frame_shape: Tuple) -> List[Dict]:
        """Process YOLO detection results"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                class_name = self.model.names[int(cls_id)]
                
                # Only interested in persons for now
                if class_name == 'person':
                    detection = {
                        'id': f'yolo_{i}',
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_name': class_name,
                        'equipment_type': 'person',
                        'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                        'detection_method': 'yolo'
                    }
                    detections.append(detection)
        
        return detections

    def _detect_safety_equipment_by_color(self, frame: np.ndarray, person_detections: List[Dict]) -> List[Dict]:
        """
        Detect safety equipment using color-based analysis
        This is a pragmatic approach until a custom model is available
        """
        safety_detections = []
        
        # Convert frame to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # For each detected person, look for safety equipment in their vicinity
        for person in person_detections:
            person_bbox = person['bbox']
            
            # Expand search area around person
            x1, y1, x2, y2 = person_bbox
            expand_ratio = 0.3  # Expand by 30%
            w, h = x2 - x1, y2 - y1
            
            # Head area (for helmet detection)
            head_x1 = max(0, int(x1 - w * expand_ratio))
            head_y1 = max(0, int(y1 - h * 0.3))
            head_x2 = min(frame.shape[1], int(x2 + w * expand_ratio))
            head_y2 = min(frame.shape[0], int(y1 + h * 0.3))
            
            # Torso area (for high-vis jacket detection)
            torso_x1 = max(0, int(x1 - w * expand_ratio))
            torso_y1 = max(0, int(y1 + h * 0.2))
            torso_x2 = min(frame.shape[1], int(x2 + w * expand_ratio))
            torso_y2 = min(frame.shape[0], int(y2 + h * 0.2))
            
            # Check for helmet in head area
            helmet_detected = self._detect_helmet_by_color(hsv, head_x1, head_y1, head_x2, head_y2)
            if helmet_detected:
                helmet_detection = {
                    'id': f'helmet_{person["id"]}',
                    'bbox': [head_x1, head_y1, head_x2, head_y2],
                    'confidence': 0.7,  # Moderate confidence for color-based detection
                    'class_name': 'helmet',
                    'equipment_type': 'helmet',
                    'center': [(head_x1 + head_x2) / 2, (head_y1 + head_y2) / 2],
                    'detection_method': 'color',
                    'associated_person': person['id']
                }
                safety_detections.append(helmet_detection)
            
            # Check for high-vis jacket in torso area
            jacket_detected = self._detect_high_vis_by_color(hsv, torso_x1, torso_y1, torso_x2, torso_y2)
            if jacket_detected:
                jacket_detection = {
                    'id': f'jacket_{person["id"]}',
                    'bbox': [torso_x1, torso_y1, torso_x2, torso_y2],
                    'confidence': 0.6,  # Moderate confidence for color-based detection
                    'class_name': 'reflective_jacket',
                    'equipment_type': 'reflective_jacket',
                    'center': [(torso_x1 + torso_x2) / 2, (torso_y1 + torso_y2) / 2],
                    'detection_method': 'color',
                    'associated_person': person['id']
                }
                safety_detections.append(jacket_detection)
        
        return safety_detections

    def _detect_helmet_by_color(self, hsv: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Detect helmet using color analysis"""
        roi = hsv[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        
        total_pixels = roi.shape[0] * roi.shape[1]
        min_color_ratio = 0.1  # At least 10% of the area should be safety color
        
        for color_name, (lower, upper) in self.safety_colors['helmet'].items():
            mask = cv2.inRange(roi, np.array(lower), np.array(upper))
            color_pixels = cv2.countNonZero(mask)
            
            if color_pixels / total_pixels > min_color_ratio:
                return True
        
        return False

    def _detect_high_vis_by_color(self, hsv: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Detect high-visibility jacket using color analysis"""
        roi = hsv[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        
        total_pixels = roi.shape[0] * roi.shape[1]
        min_color_ratio = 0.15  # At least 15% of the area should be high-vis color
        
        for color_name, (lower, upper) in self.safety_colors['high_vis'].items():
            mask = cv2.inRange(roi, np.array(lower), np.array(upper))
            color_pixels = cv2.countNonZero(mask)
            
            if color_pixels / total_pixels > min_color_ratio:
                return True
        
        return False

    def _analyze_safety_compliance(self, detections: List[Dict], frame: np.ndarray) -> Dict:
        """Analyze safety compliance based on detections"""
        persons = [d for d in detections if d['equipment_type'] == 'person']
        helmets = [d for d in detections if d['equipment_type'] == 'helmet']
        jackets = [d for d in detections if d['equipment_type'] == 'reflective_jacket']
        
        total_persons = len(persons)
        persons_with_helmets = 0
        persons_with_jackets = 0
        
        # Count compliance using association
        for person in persons:
            # Check for helmet association
            helmet_found = any(
                h.get('associated_person') == person['id'] 
                for h in helmets
            )
            
            # Check for jacket association
            jacket_found = any(
                j.get('associated_person') == person['id'] 
                for j in jackets
            )
            
            if helmet_found:
                persons_with_helmets += 1
            if jacket_found:
                persons_with_jackets += 1
        
        # Calculate compliance rates
        helmet_compliance = (persons_with_helmets / total_persons * 100) if total_persons > 0 else 100
        jacket_compliance = (persons_with_jackets / total_persons * 100) if total_persons > 0 else 100
        overall_compliance = min(helmet_compliance, jacket_compliance)
        
        # More lenient compliance threshold for color-based detection
        COMPLIANCE_THRESHOLD = 60.0
        
        violations = []
        missing_helmets = total_persons - persons_with_helmets
        missing_jackets = total_persons - persons_with_jackets
        
        if missing_helmets > 0:
            violations.append(f'{missing_helmets} worker(s) possibly missing helmet')
        if missing_jackets > 0:
            violations.append(f'{missing_jackets} worker(s) possibly missing high-vis jacket')
        
        is_compliant = overall_compliance >= COMPLIANCE_THRESHOLD
        
        return {
            'total_persons': total_persons,
            'persons_with_helmets': persons_with_helmets,
            'persons_with_jackets': persons_with_jackets,
            'helmet_compliance_rate': helmet_compliance,
            'jacket_compliance_rate': jacket_compliance,
            'overall_compliance_rate': overall_compliance,
            'violations': violations,
            'is_compliant': is_compliant,
            'detection_note': 'Using color-based detection - train custom model for better accuracy'
        }

    def draw_detections(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw detection results on frame"""
        if 'error' in results:
            # Draw error message
            cv2.putText(frame, f"Error: {results['error']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        output_frame = frame.copy()
        detections = results['detections']
        safety_analysis = results['safety_analysis']
        
        # Color scheme
        colors = {
            'person': (0, 255, 0),      # Green
            'helmet': (0, 0, 255),      # Red
            'reflective_jacket': (255, 0, 0),  # Blue
            'other': (128, 128, 128)    # Gray
        }
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            equipment_type = detection['equipment_type']
            confidence = detection['confidence']
            method = detection.get('detection_method', 'unknown')
            
            # Draw bounding box
            color = colors.get(equipment_type, (128, 128, 128))
            cv2.rectangle(output_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw label with detection method
            label = f"{equipment_type}: {confidence:.2f} ({method})"
            cv2.putText(output_frame, label,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw safety status
        self._draw_safety_status(output_frame, safety_analysis)
        
        return output_frame

    def _draw_safety_status(self, frame: np.ndarray, safety_analysis: Dict):
        """Draw safety compliance status on frame"""
        height, width = frame.shape[:2]
        
        # Status background
        status_bg = (0, 0, 0)  # Black background
        status_color = (0, 255, 0) if safety_analysis['is_compliant'] else (0, 0, 255)
        
        # Draw status box
        cv2.rectangle(frame, (10, 10), (450, 140), status_bg, -1)
        cv2.rectangle(frame, (10, 10), (450, 140), status_color, 2)
        
        # Status text
        status_text = "COMPLIANT" if safety_analysis['is_compliant'] else "VIOLATION"
        cv2.putText(frame, f"Safety Status: {status_text}",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Compliance rates
        cv2.putText(frame, f"Helmet Compliance: {safety_analysis['helmet_compliance_rate']:.1f}%",
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Jacket Compliance: {safety_analysis['jacket_compliance_rate']:.1f}%",
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Worker count
        cv2.putText(frame, f"Workers Detected: {safety_analysis['total_persons']}",
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection method note
        cv2.putText(frame, "Color-based detection (train custom model for accuracy)",
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Violations
        if safety_analysis['violations']:
            cv2.putText(frame, f"Violations: {len(safety_analysis['violations'])}",
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def process_video_stream(self, video_source: str = 0, save_output: bool = False, output_path: str = None):
        """Process live video stream for safety detection"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if saving output
        writer = None
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.logger.info(f"Starting video processing: {width}x{height} @ {fps} FPS")
        self.logger.info("Using color-based safety equipment detection")
        self.logger.info("For better accuracy, train a custom model with safety equipment dataset")
        
        try:
            frame_count = 0
            violation_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Detect safety equipment
                results = self.detect_safety_equipment(frame)
                
                # Draw results
                output_frame = self.draw_detections(frame, results)
                
                # Track violations
                if 'safety_analysis' in results and not results['safety_analysis']['is_compliant']:
                    violation_frames += 1
                
                # Save frame if requested
                if writer:
                    writer.write(output_frame)
                
                # Display frame
                cv2.imshow('Safety Detection System (Color-Based)', output_frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Print statistics every 100 frames
                if frame_count % 100 == 0:
                    violation_rate = (violation_frames / frame_count) * 100
                    self.logger.info(f"Processed {frame_count} frames, violation rate: {violation_rate:.1f}%")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            violation_rate = (violation_frames / frame_count) * 100 if frame_count > 0 else 0
            self.logger.info(f"Processing complete. Total frames: {frame_count}, Violation rate: {violation_rate:.1f}%")


# Example usage
if __name__ == "__main__":
    # Initialize fixed safety detector
    detector = SafetyDetectorFixed(confidence_threshold=0.3)
    
    # Test with webcam
    try:
        detector.process_video_stream(
            video_source=0,
            save_output=True,
            output_path="safety_detection_fixed_output.mp4"
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a camera connected")
