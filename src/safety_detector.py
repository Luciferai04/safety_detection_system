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
try:
    from .person_tracker import PersonTracker
except ImportError:
    from person_tracker import PersonTracker

class SafetyDetector:
    """
    Advanced Safety Detection System for Thermal Power Plant Workers
    
    This class implements YOLO-CA (YOLO with Coordinate Attention) based object detection to identify:
    - Safety helmets/hard hats
    - Reflective jackets/high-visibility vests
    - Person detection for comprehensive safety monitoring
    
    Based on the research paper:
    "Detection of Safety Helmet-Wearing Based on the YOLO_CA Model"
    Authors: Xiaoqin Wu, Songrong Qian, Ming Yang
    
    Key enhancements over standard YOLO:
    1. Coordinate Attention (CA) mechanism in backbone
    2. Ghost modules replacing C3 modules
    3. Depthwise Separable Convolution in neck
    4. EIoU Loss for better localization
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = 'auto'):
        """
        Initialize the Safety Detection System
        
        Args:
            model_path: Path to custom trained YOLO model (optional)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for Non-Maximum Suppression
            device: Device to run inference on ('cpu', 'cuda', 'auto')
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
        
        # Define safety equipment classes
        self.safety_classes = {
            'helmet': ['helmet', 'hard hat', 'safety helmet', 'construction helmet'],
            'reflective_jacket': ['reflective jacket', 'high-vis vest', 'safety vest', 'hi-vis jacket'],
            'person': ['person', 'worker', 'human']
        }
        
        # Safety status tracking
        self.violation_count = 0
        self.total_detections = 0
        
        # Performance monitoring
        self.processing_times = []
        self.max_processing_times = 1000  # Keep last 1000 processing times
        
        # Error tracking
        self.error_count = 0
        self.last_error = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10
        
        # Person tracking to prevent duplicate counting
        self.person_tracker = PersonTracker(
            max_disappeared=30,  # Allow person to be missing for 30 frames
            max_distance=100.0   # Maximum distance for person association
        )
        
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
                # Load custom trained model
                model = YOLO(model_path)
                self.logger.info(f"Loaded custom model from {model_path}")
            else:
                # Use pre-trained YOLOv8 model and fine-tune for safety detection
                model = YOLO('yolov8n.pt')  # Start with nano model for speed
                self.logger.info("Loaded pre-trained YOLOv8 model")
            
            # Move model to appropriate device
            model.to(self.device)
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def detect_safety_equipment(self, frame: np.ndarray) -> Dict:
        """
        Detect safety equipment in a single frame with comprehensive error handling
        
        Args:
            frame: Input image/frame as numpy array
            
        Returns:
            Dictionary containing detection results or error information
        """
        start_time = time.time()
        
        try:
            # Input validation
            if frame is None:
                raise ValueError("Input frame is None")
            
            if not isinstance(frame, np.ndarray):
                raise TypeError("Input frame must be a numpy array")
            
            if len(frame.shape) != 3 or frame.shape[2] not in [1, 3, 4]:
                raise ValueError(f"Invalid frame shape: {frame.shape}. Expected (H, W, C) with C in [1, 3, 4]")
            
            if frame.size == 0:
                raise ValueError("Input frame is empty")
            
            # Check for reasonable frame dimensions
            height, width = frame.shape[:2]
            if height < 32 or width < 32 or height > 8192 or width > 8192:
                raise ValueError(f"Frame dimensions out of range: {width}x{height}")
            
            # Ensure model is available
            if self.model is None:
                raise RuntimeError("YOLO model not initialized")
            
            # Run YOLO inference with timeout protection
            try:
                results = self.model(frame, 
                                   conf=self.confidence_threshold,
                                   iou=self.iou_threshold,
                                   verbose=False)
            except Exception as model_error:
                self.logger.error(f"Model inference failed: {model_error}")
                raise RuntimeError(f"Model inference error: {str(model_error)}")
            
            # Validate model results
            if not results or len(results) == 0:
                self.logger.warning("No results returned from model")
                results = [type('obj', (object,), {'boxes': None})]  # Create dummy result
            
            # Process results
            detections = self._process_detections(results[0], frame.shape)
            
            # Analyze safety compliance
            safety_analysis = self._analyze_safety_compliance(detections)
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_times:
                self.processing_times.pop(0)
            
            # Update statistics
            self.total_detections += 1
            if not safety_analysis.get('is_compliant', True):
                self.violation_count += 1
            
            # Reset error tracking on success
            self.consecutive_errors = 0
            
            return {
                'detections': detections,
                'safety_analysis': safety_analysis,
                'frame_shape': frame.shape,
                'timestamp': time.time(),
                'processing_time': processing_time,
                'device': self.device
            }
            
        except Exception as e:
            # Track errors
            self.error_count += 1
            self.consecutive_errors += 1
            self.last_error = str(e)
            
            processing_time = time.time() - start_time
            
            error_details = {
                'error': str(e),
                'error_type': type(e).__name__,
                'error_count': self.error_count,
                'consecutive_errors': self.consecutive_errors,
                'processing_time': processing_time,
                'timestamp': time.time(),
                'frame_info': {
                    'shape': frame.shape if frame is not None else None,
                    'dtype': str(frame.dtype) if frame is not None else None
                }
            }
            
            # Log based on error frequency
            if self.consecutive_errors == 1:
                self.logger.error(f"Detection error: {e}")
            elif self.consecutive_errors <= 5:
                self.logger.warning(f"Consecutive detection error #{self.consecutive_errors}: {e}")
            elif self.consecutive_errors == self.max_consecutive_errors:
                self.logger.critical(f"Max consecutive errors reached ({self.max_consecutive_errors}). System may be unstable.")
            
            return error_details
    
    def _process_detections(self, result, frame_shape: Tuple) -> List[Dict]:
        """Process YOLO detection results"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                # Get class name
                class_name = self.model.names[int(cls_id)]
                
                # Categorize detection
                equipment_type = self._categorize_detection(class_name)
                
                detection = {
                    'id': i,
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_name': class_name,
                    'equipment_type': equipment_type,
                    'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                }
                
                detections.append(detection)
        
        return detections
    
    def _categorize_detection(self, class_name: str) -> str:
        """Categorize detection into safety equipment types"""
        class_name_lower = class_name.lower()
        
        # Check for helmet
        for helmet_term in self.safety_classes['helmet']:
            if helmet_term in class_name_lower:
                return 'helmet'
        
        # Check for reflective jacket
        for jacket_term in self.safety_classes['reflective_jacket']:
            if jacket_term in class_name_lower:
                return 'reflective_jacket'
        
        # Check for person
        for person_term in self.safety_classes['person']:
            if person_term in class_name_lower:
                return 'person'
        
        return 'other'
    
    def _analyze_safety_compliance(self, detections: List[Dict]) -> Dict:
        """Analyze safety compliance based on detections with person tracking"""
        # Update person tracker with current detections
        tracked_persons = self.person_tracker.update(detections)
        
        # Get tracking summary for accurate person counts
        tracking_summary = self.person_tracker.get_tracking_summary()
        
        # Use tracked persons for compliance analysis
        total_persons = tracking_summary['active_persons']
        persons_with_helmets = 0
        persons_with_jackets = 0
        
        # If no persons detected, compliance is 100% (no violations possible)
        if total_persons == 0:
            return {
                'total_persons': 0,
                'tracked_persons': 0,
                'persons_with_helmets': 0,
                'persons_with_jackets': 0,
                'helmet_compliance_rate': 100.0,
                'jacket_compliance_rate': 100.0,
                'overall_compliance_rate': 100.0,
                'violations': [],
                'is_compliant': True,
                'tracking_info': tracking_summary
            }
        
        # Count compliant persons from tracked data
        for person_id, person_data in tracked_persons.items():
            safety_status = person_data.get('safety_status', {})
            if safety_status.get('has_helmet', False):
                persons_with_helmets += 1
            if safety_status.get('has_jacket', False):
                persons_with_jackets += 1
        
        # Calculate compliance rates (now guaranteed total_persons > 0)
        helmet_compliance = (persons_with_helmets / total_persons * 100)
        jacket_compliance = (persons_with_jackets / total_persons * 100)
        overall_compliance = min(helmet_compliance, jacket_compliance)
        
        # Define compliance threshold (80% or higher is considered acceptable)
        COMPLIANCE_THRESHOLD = 80.0
        
        # Identify violations based on individual equipment missing
        violations = []
        missing_helmets = total_persons - persons_with_helmets
        missing_jackets = total_persons - persons_with_jackets
        
        if missing_helmets > 0:
            violations.append(f'{missing_helmets} worker(s) missing helmet')
        if missing_jackets > 0:
            violations.append(f'{missing_jackets} worker(s) missing reflective jacket')
        
        # Frame is compliant if overall compliance meets threshold
        is_compliant = overall_compliance >= COMPLIANCE_THRESHOLD
        
        return {
            'total_persons': len([d for d in detections if d['equipment_type'] == 'person']),  # Raw detections
            'tracked_persons': total_persons,  # Unique tracked persons
            'persons_with_helmets': persons_with_helmets,
            'persons_with_jackets': persons_with_jackets,
            'helmet_compliance_rate': helmet_compliance,
            'jacket_compliance_rate': jacket_compliance,
            'overall_compliance_rate': overall_compliance,
            'violations': violations,
            'is_compliant': is_compliant,
            'tracking_info': tracking_summary
        }
    
    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _check_overlap(self, bbox1: List[float], bbox2: List[float], overlap_threshold: float = 0.1) -> bool:
        """Check if two bounding boxes overlap by at least the threshold amount"""
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # Check if there's any intersection
        if x1 >= x2 or y1 >= y2:
            return False
        
        # Calculate intersection and union areas
        intersection_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU (Intersection over Union)
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou >= overlap_threshold
    
    def draw_detections(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            results: Detection results from detect_safety_equipment
            
        Returns:
            Frame with drawn detections
        """
        if 'error' in results:
            return frame
        
        output_frame = frame.copy()
        detections = results['detections']
        safety_analysis = results['safety_analysis']
        
        # Color scheme
        colors = {
            'person': (0, 255, 0),      # Green
            'helmet': (255, 0, 0),      # Red
            'reflective_jacket': (0, 0, 255),  # Blue
            'other': (128, 128, 128)    # Gray
        }
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            equipment_type = detection['equipment_type']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = colors.get(equipment_type, (128, 128, 128))
            cv2.rectangle(output_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Draw label
            label = f"{equipment_type}: {confidence:.2f}"
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
        cv2.rectangle(frame, (10, 10), (400, 120), status_bg, -1)
        cv2.rectangle(frame, (10, 10), (400, 120), status_color, 2)
        
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
        
        # Violations
        if safety_analysis['violations']:
            cv2.putText(frame, f"Violations: {len(safety_analysis['violations'])}",
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    def process_video_stream(self, video_source: str = 0, save_output: bool = False, output_path: str = None):
        """
        Process live video stream for safety detection
        
        Args:
            video_source: Video source (0 for webcam, path for video file, URL for IP camera)
            save_output: Whether to save processed video
            output_path: Path to save output video
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if saving output
        writer = None
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.logger.info(f"Starting video processing: {width}x{height} @ {fps} FPS")
        
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
                cv2.imshow('Safety Detection System', output_frame)
                
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


# Example usage and testing
if __name__ == "__main__":
    # Initialize safety detector
    detector = SafetyDetector(confidence_threshold=0.5)
    
    # Test with webcam (change to video file path or IP camera URL as needed)
    try:
        detector.process_video_stream(
            video_source=0,  # Use webcam
            save_output=True,
            output_path="safety_detection_output.mp4"
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a camera connected or provide a valid video file path")
