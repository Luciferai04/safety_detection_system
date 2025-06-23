#!/usr/bin/env python3
"""
Production-ready REST API for Safety Detection System

This module provides a secure and robust Flask API for the thermal power plant
safety detection system with proper authentication, rate limiting, and error handling.
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import threading
import time
import secrets
from datetime import datetime
import os
import logging
from werkzeug.utils import secure_filename
from functools import wraps
import uuid

from safety_detector import SafetyDetector
from config_manager import get_config_manager
from camera_manager import CameraManager

# Initialize configuration
config = get_config_manager()

app = Flask(__name__)

# Security configurations
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['UPLOAD_FOLDER'] = config.raw_config.get('api', {}).get('upload_folder', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = config.security.max_file_size_mb * 1024 * 1024

# CORS with restricted origins
CORS(app, origins=config.security.allowed_origins)

# Setup logging with error handling
try:
 # Ensure logs directory exists
 log_dir = os.path.dirname(config.logging.file_path)
 if log_dir:
 os.makedirs(log_dir, exist_ok=True)

 logging.basicConfig(
 level=getattr(logging, config.logging.level.upper()),
 format=config.logging.log_format,
 handlers=[
 logging.FileHandler(config.logging.file_path),
 logging.StreamHandler()
 ]
 )
except Exception as e:
 # Fallback to console logging only if file logging fails
 logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
 handlers=[logging.StreamHandler()]
 )
 print(f"Warning: Could not setup file logging: {e}")
logger = logging.getLogger(__name__)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# File type validation
ALLOWED_EXTENSIONS = set(config.security.allowed_file_types)

# API Key authentication decorator
def require_api_key(f):
 @wraps(f)
 def decorated_function(*args, **kwargs):
 if not config.security.enable_api_key_auth:
 return f(*args, **kwargs)

 api_key = request.headers.get(config.security.api_key_header)
 expected_key = config.get_api_key()

 if not api_key or not expected_key or api_key != expected_key:
 logger.warning(f"Unauthorized API access attempt from {request.remote_addr}")
 return jsonify({'error': 'Invalid or missing API key'}), 401

 return f(*args, **kwargs)
 return decorated_function

# Global detector instance
detector = SafetyDetector(confidence_threshold=0.5)

# Global camera manager instance
camera_manager = CameraManager()

# Global variables for video streaming
video_feed_active = False
current_frame = None
frame_lock = threading.Lock()

def allowed_file(filename):
 """Check if file extension is allowed"""
 return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image):
 """Convert OpenCV image to base64 string"""
 _, buffer = cv2.imencode('.jpg', image)
 image_base64 = base64.b64encode(buffer).decode('utf-8')
 return image_base64

def decode_base64_to_image(base64_string):
 """Convert base64 string to OpenCV image"""
 image_data = base64.b64decode(base64_string)
 image = Image.open(io.BytesIO(image_data))
 image_array = np.array(image)
 if len(image_array.shape) == 3 and image_array.shape[2] == 3:
 image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
 return image_array

@app.route('/api/health', methods=['GET'])
def health_check():
 """Health check endpoint"""
 return jsonify({
 'status': 'healthy',
 'timestamp': datetime.now().isoformat(),
 'detector_ready': detector is not None,
 'device': detector.device if detector else 'unknown'
 })

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
 """Detect safety equipment in uploaded image"""
 try:
 if 'image' not in request.files:
 return jsonify({'error': 'No image file provided'}), 400

 file = request.files['image']
 if file.filename == '':
 return jsonify({'error': 'No image file selected'}), 400

 if file and allowed_file(file.filename):
 # Read image
 image_stream = io.BytesIO(file.read())
 image = Image.open(image_stream)
 image_array = np.array(image)

 # Convert RGB to BGR for OpenCV
 if len(image_array.shape) == 3 and image_array.shape[2] == 3:
 image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

 # Detect safety equipment
 results = detector.detect_safety_equipment(image_array)

 # Draw detections on image
 output_image = detector.draw_detections(image_array, results)

 # Convert to base64 for response
 output_image_base64 = encode_image_to_base64(output_image)

 response = {
 'success': True,
 'results': results,
 'output_image': output_image_base64,
 'timestamp': datetime.now().isoformat()
 }

 return jsonify(response)

 else:
 return jsonify({'error': 'Invalid file type'}), 400

 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/detect/base64', methods=['POST'])
def detect_base64_image():
 """Detect safety equipment in base64 encoded image"""
 try:
 data = request.get_json()
 if 'image' not in data:
 return jsonify({'error': 'No image data provided'}), 400

 # Decode base64 image
 image_array = decode_base64_to_image(data['image'])

 # Detect safety equipment
 results = detector.detect_safety_equipment(image_array)

 # Draw detections if requested
 include_output_image = data.get('include_output_image', False)
 response = {
 'success': True,
 'results': results,
 'timestamp': datetime.now().isoformat()
 }

 if include_output_image:
 output_image = detector.draw_detections(image_array, results)
 response['output_image'] = encode_image_to_base64(output_image)

 return jsonify(response)

 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/detect/video', methods=['POST'])
def detect_video():
 """Analyze uploaded video file for safety compliance"""
 try:
 if 'video' not in request.files:
 return jsonify({'error': 'No video file provided'}), 400

 file = request.files['video']
 if file.filename == '':
 return jsonify({'error': 'No video file selected'}), 400

 if file and allowed_file(file.filename):
 # Save uploaded file
 filename = secure_filename(file.filename)
 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
 file.save(filepath)

 # Process video
 cap = cv2.VideoCapture(filepath)
 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 fps = cap.get(cv2.CAP_PROP_FPS)

 results_data = []
 frame_count = 0
 violation_frames = 0

 # Process every 30th frame for efficiency
 frame_skip = 30

 while True:
 ret, frame = cap.read()
 if not ret:
 break

 frame_count += 1

 if frame_count % frame_skip == 0:
 # Detect safety equipment
 results = detector.detect_safety_equipment(frame)

 if 'safety_analysis' in results:
 analysis = results['safety_analysis']

 frame_data = {
 'frame_number': frame_count,
 'timestamp': frame_count / fps,
 'total_persons': analysis['total_persons'],
 'persons_with_helmets': analysis['persons_with_helmets'],
 'persons_with_jackets': analysis['persons_with_jackets'],
 'helmet_compliance_rate': analysis['helmet_compliance_rate'],
 'jacket_compliance_rate': analysis['jacket_compliance_rate'],
 'overall_compliance_rate': analysis['overall_compliance_rate'],
 'violations': analysis['violations'],
 'is_compliant': analysis['is_compliant']
 }

 results_data.append(frame_data)

 if not analysis['is_compliant']:
 violation_frames += 1

 cap.release()

 # Clean up uploaded file
 os.remove(filepath)

 # Calculate summary statistics
 total_analyzed_frames = len(results_data)
 avg_helmet_compliance = np.mean([r['helmet_compliance_rate'] for r in results_data]) if results_data else 0
 avg_jacket_compliance = np.mean([r['jacket_compliance_rate'] for r in results_data]) if results_data else 0
 violation_rate = (violation_frames / total_analyzed_frames * 100) if total_analyzed_frames > 0 else 0

 summary = {
 'total_frames': total_frames,
 'analyzed_frames': total_analyzed_frames,
 'violation_frames': violation_frames,
 'violation_rate': violation_rate,
 'average_helmet_compliance': avg_helmet_compliance,
 'average_jacket_compliance': avg_jacket_compliance,
 'video_duration': total_frames / fps if fps > 0 else 0
 }

 response = {
 'success': True,
 'summary': summary,
 'frame_analysis': results_data,
 'timestamp': datetime.now().isoformat()
 }

 return jsonify(response)

 else:
 return jsonify({'error': 'Invalid file type'}), 400

 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/stream/start', methods=['POST'])
def start_video_stream():
 """Start video stream processing"""
 global video_feed_active

 try:
 data = request.get_json()
 camera_index = data.get('camera_index', 0)

 if video_feed_active:
 return jsonify({'error': 'Video stream already active'}), 400

 # Start video capture in separate thread
 def video_capture_thread():
 global video_feed_active, current_frame

 cap = cv2.VideoCapture(camera_index)
 video_feed_active = True

 while video_feed_active:
 ret, frame = cap.read()
 if not ret:
 break

 # Detect safety equipment
 results = detector.detect_safety_equipment(frame)
 output_frame = detector.draw_detections(frame, results)

 with frame_lock:
 current_frame = output_frame.copy()

 time.sleep(0.1) # Limit to ~10 FPS

 cap.release()

 thread = threading.Thread(target=video_capture_thread)
 thread.daemon = True
 thread.start()

 return jsonify({
 'success': True,
 'message': 'Video stream started',
 'camera_index': camera_index
 })

 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/stream/stop', methods=['POST'])
def stop_video_stream():
 """Stop video stream processing"""
 global video_feed_active

 video_feed_active = False

 return jsonify({
 'success': True,
 'message': 'Video stream stopped'
 })

@app.route('/api/stream/frame', methods=['GET'])
def get_current_frame():
 """Get current frame from video stream"""
 global current_frame

 try:
 with frame_lock:
 if current_frame is not None:
 frame_base64 = encode_image_to_base64(current_frame)
 return jsonify({
 'success': True,
 'frame': frame_base64,
 'timestamp': datetime.now().isoformat()
 })
 else:
 return jsonify({'error': 'No frame available'}), 404

 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/cameras/detect', methods=['GET'])
def detect_cameras():
 """Detect available cameras on the system"""
 try:
 cameras = camera_manager.detect_cameras()
 return jsonify({
 'success': True,
 'cameras': cameras,
 'timestamp': datetime.now().isoformat()
 })
 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/cameras/test', methods=['POST'])
def test_camera():
 """Test camera connection"""
 try:
 data = request.get_json()
 camera_source = data.get('camera_source')
 timeout = data.get('timeout', 5)

 if camera_source is None:
 return jsonify({'error': 'camera_source is required'}), 400

 success, properties = camera_manager.test_camera_connection(camera_source, timeout)

 return jsonify({
 'success': success,
 'properties': properties,
 'timestamp': datetime.now().isoformat()
 })
 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/cameras/validate', methods=['POST'])
def validate_camera():
 """Validate camera configuration"""
 try:
 data = request.get_json()

 config = camera_manager.create_camera_source(
 camera_type=data.get('camera_type', 'webcam'),
 source=data.get('source'),
 username=data.get('username'),
 password=data.get('password')
 )

 is_valid, message = camera_manager.validate_camera_source(config)

 return jsonify({
 'success': True,
 'is_valid': is_valid,
 'message': message,
 'config': config,
 'timestamp': datetime.now().isoformat()
 })
 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/cameras/discover', methods=['POST'])
def discover_cameras():
 """Auto-discover IP cameras on network"""
 try:
 data = request.get_json()
 ip_range = data.get('ip_range', '192.168.1.')

 discovered = camera_manager.auto_discover_cameras(ip_range)

 return jsonify({
 'success': True,
 'discovered_cameras': discovered,
 'timestamp': datetime.now().isoformat()
 })
 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/cameras/urls', methods=['GET'])
def get_camera_urls():
 """Get common camera URL patterns"""
 try:
 url_patterns = camera_manager.get_common_camera_urls()
 return jsonify({
 'success': True,
 'url_patterns': url_patterns,
 'timestamp': datetime.now().isoformat()
 })
 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def detector_config():
 """Get or update detector configuration"""
 global detector

 if request.method == 'GET':
 return jsonify({
 'confidence_threshold': detector.confidence_threshold,
 'iou_threshold': detector.iou_threshold,
 'device': detector.device
 })

 elif request.method == 'POST':
 try:
 data = request.get_json()

 # Update configuration
 if 'confidence_threshold' in data:
 detector.confidence_threshold = data['confidence_threshold']

 if 'iou_threshold' in data:
 detector.iou_threshold = data['iou_threshold']

 return jsonify({
 'success': True,
 'message': 'Configuration updated',
 'config': {
 'confidence_threshold': detector.confidence_threshold,
 'iou_threshold': detector.iou_threshold,
 'device': detector.device
 }
 })

 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
 """Get system statistics"""
 return jsonify({
 'violation_count': detector.violation_count,
 'total_detections': detector.total_detections,
 'system_uptime': time.time(),
 'video_stream_active': video_feed_active
 })

@app.route('/api/docs', methods=['GET'])
def api_documentation():
 """API documentation"""
 docs = {
 'title': 'Safety Detection System API',
 'version': '1.0.0',
 'description': 'API for detecting safety equipment in thermal power plant environments',
 'endpoints': {
 '/api/health': {
 'method': 'GET',
 'description': 'Health check endpoint',
 'response': 'System status and detector information'
 },
 '/api/detect/image': {
 'method': 'POST',
 'description': 'Upload image file for safety detection',
 'parameters': 'multipart/form-data with image file',
 'response': 'Detection results and annotated image'
 },
 '/api/detect/base64': {
 'method': 'POST',
 'description': 'Send base64 encoded image for detection',
 'parameters': 'JSON with base64 image data',
 'response': 'Detection results and optional annotated image'
 },
 '/api/detect/video': {
 'method': 'POST',
 'description': 'Upload video file for analysis',
 'parameters': 'multipart/form-data with video file',
 'response': 'Frame-by-frame analysis and summary statistics'
 },
 '/api/stream/start': {
 'method': 'POST',
 'description': 'Start live video stream processing',
 'parameters': 'JSON with camera_index (optional)',
 'response': 'Stream status'
 },
 '/api/stream/stop': {
 'method': 'POST',
 'description': 'Stop live video stream processing',
 'response': 'Stream status'
 },
 '/api/stream/frame': {
 'method': 'GET',
 'description': 'Get current frame from video stream',
 'response': 'Base64 encoded current frame'
 },
 '/api/config': {
 'method': 'GET/POST',
 'description': 'Get or update detector configuration',
 'response': 'Current configuration settings'
 }
 }
 }

 return jsonify(docs)


# Enhanced detector integration
try:
 from enhanced_safety_detector import EnhancedSafetyDetector
 enhanced_detector_global = EnhancedSafetyDetector(confidence_threshold=0.6)
 ENHANCED_API_AVAILABLE = True
 print(" Enhanced API detector loaded")
except Exception as e:
 enhanced_detector_global = None
 ENHANCED_API_AVAILABLE = False
 print(f"Enhanced detector not available for API: {e}")

@app.route('/api/detect/enhanced', methods=['POST'])
def detect_enhanced():
 """Enhanced detection with area-specific rules"""
 try:
 if not ENHANCED_API_AVAILABLE:
 return jsonify({'error': 'Enhanced detection not available'}), 503

 if 'image' not in request.files:
 return jsonify({'error': 'No image file provided'}), 400

 file = request.files['image']
 area = request.form.get('area', 'general')

 if file.filename == '':
 return jsonify({'error': 'No image file selected'}), 400

 if file and allowed_file(file.filename):
 # Read image
 image_stream = io.BytesIO(file.read())
 image = Image.open(image_stream)
 image_array = np.array(image)

 # Convert RGB to BGR for OpenCV
 if len(image_array.shape) == 3 and image_array.shape[2] == 3:
 image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

 # Enhanced detection
 results = enhanced_detector_global.detect_enhanced_safety_equipment(
 image_array,
 area=area,
 environmental_conditions=None
 )

 # Draw enhanced results
 output_image = enhanced_detector_global.draw_enhanced_detections(image_array, results)
 output_image_base64 = encode_image_to_base64(output_image)

 response = {
 'success': True,
 'enhanced_results': results,
 'output_image': output_image_base64,
 'area': area,
 'timestamp': datetime.now().isoformat(),
 'features': {
 'environmental_processing': True,
 'area_specific_rules': True,
 'critical_ppe_detection': True,
 'enhanced_accuracy': True
 }
 }

 return jsonify(response)
 else:
 return jsonify({'error': 'Invalid file type'}), 400

 except Exception as e:
 return jsonify({'error': str(e)}), 500

@app.route('/api/areas', methods=['GET'])
def get_thermal_plant_areas():
 """Get available thermal plant areas"""
 if ENHANCED_API_AVAILABLE:
 areas = enhanced_detector_global.area_ppe_requirements
 return jsonify({
 'success': True,
 'areas': areas,
 'timestamp': datetime.now().isoformat()
 })
 else:
 return jsonify({'error': 'Enhanced detection not available'}), 503

if __name__ == '__main__':
 print(" Starting Safety Detection API Server...")
 print(" API Documentation available at: http://localhost:5000/api/docs")
 app.run(debug=True, host='0.0.0.0', port=5000)
