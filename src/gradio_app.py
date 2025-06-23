#!/usr/bin/env python3
"""
Gradio Web Interface for Safety Detection System

This module provides a modern web interface using Gradio for the thermal power plant
safety detection system with live camera support and real-time analysis.
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import time
import threading
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import base64
import io
from typing import Optional, Tuple, Dict, Any

from safety_detector import SafetyDetector
from config_manager import get_config_manager
from camera_manager import CameraManager

# Global variables for live camera feed
camera_active = False
current_detector = None
detection_history = []
violation_stats = {"total": 0, "helmet": 0, "jacket": 0}
live_camera_thread = None
current_frame = None
current_annotated_frame = None
camera_manager = CameraManager()
current_camera_source = 0
available_cameras = []

# Configuration
config = get_config_manager()


# Enhanced detector integration
try:
 from .enhanced_safety_detector import EnhancedSafetyDetector
 ENHANCED_DETECTOR_AVAILABLE = True
except ImportError:
 from enhanced_safety_detector import EnhancedSafetyDetector
 ENHANCED_DETECTOR_AVAILABLE = True
except Exception as e:
 print(f"Enhanced detector not available: {e}")
 ENHANCED_DETECTOR_AVAILABLE = False

# Initialize enhanced detector
if ENHANCED_DETECTOR_AVAILABLE:
 enhanced_detector = EnhancedSafetyDetector(confidence_threshold=0.6)
 print(" Enhanced detector loaded with 7 PPE classes")
else:
 enhanced_detector = None

def process_image_enhanced(image, area_selection="general"):
 """Enhanced image processing with area-specific rules"""
 if image is None:
 return None, "No image provided", "{}"

 try:
 # Use enhanced detector if available
 if enhanced_detector:
 results = enhanced_detector.detect_enhanced_safety_equipment(
 image,
 area=area_selection,
 environmental_conditions=None # Auto-detect
 )

 # Draw enhanced results
 output_image = enhanced_detector.draw_enhanced_detections(image, results)

 # Create enhanced analysis text
 analysis = results.get('enhanced_safety_analysis', {})
 env_conditions = results.get('environmental_conditions', [])

 analysis_text = f"""
 **Enhanced Safety Analysis - {area_selection.replace('_', ' ').title()}**

 **Detection Summary:**
- Total Persons: {analysis.get('total_persons', 0)}
- Safety Score: {analysis.get('safety_score', 0):.1f}%
- Overall Compliance: {analysis.get('overall_compliance_rate', 0):.1f}%

 **PPE Compliance Rates:**
- Helmets: {analysis.get('helmet_compliance_rate', 0):.1f}%
- Reflective Jackets: {analysis.get('reflective_jacket_compliance_rate', 0):.1f}%
- Safety Boots: {analysis.get('safety_boots_compliance_rate', 0):.1f}%
- Safety Gloves: {analysis.get('safety_gloves_compliance_rate', 0):.1f}%
- Arc Flash Suits: {analysis.get('arc_flash_suit_compliance_rate', 0):.1f}%
- Respirators: {analysis.get('respirator_compliance_rate', 0):.1f}%

 **Environmental Conditions:** {', '.join(env_conditions) if env_conditions else 'Normal'}

 **Critical Violations:** {len(analysis.get('critical_violations', []))}

 **Status:** {'COMPLIANT' if analysis.get('is_compliant', True) else 'VIOLATIONS DETECTED'}
"""

 return output_image, analysis_text, json.dumps(results, indent=2)
 else:
 # Fallback to original detector
 return process_image(image)

 except Exception as e:
 return image, f"Enhanced processing error: {str(e)}", "{}"

# Enhanced area selection dropdown
thermal_plant_areas = [
 ("General Area", "general"),
 ("Boiler Area", "boiler_area"),
 ("Switchyard (Critical)", "switchyard"),
 ("Coal Handling", "coal_handling"),
 ("Turbine Hall", "turbine_hall"),
 ("Control Room", "control_room")
]

def initialize_detector() -> SafetyDetector:
 """Initialize the safety detector"""
 global current_detector
 if current_detector is None:
 current_detector = SafetyDetector(
 confidence_threshold=config.model.confidence_threshold,
 iou_threshold=config.model.iou_threshold,
 device=config.model.device
 )
 return current_detector

def process_image(image: np.ndarray) -> Tuple[np.ndarray, str, str]:
 """
 Process a single image for safety detection

 Args:
 image: Input image as numpy array

 Returns:
 Tuple of (annotated_image, analysis_text, statistics_json)
 """
 if image is None:
 return None, "No image provided", "{}"

 detector = initialize_detector()

 try:
 # Run detection
 results = detector.detect_safety_equipment(image)

 if 'error' in results:
 return image, f"Error: {results['error']}", "{}"

 # Draw detections
 annotated_image = detector.draw_detections(image, results)

 # Get analysis
 analysis = results.get('safety_analysis', {})
 detections = results.get('detections', [])

 # Create analysis text
 analysis_text = f"""
 **Detection Results:**
- Objects Detected: {len(detections)}
- Persons: {analysis.get('total_persons', 0)}
- Persons with Helmets: {analysis.get('persons_with_helmets', 0)}
- Persons with Jackets: {analysis.get('persons_with_jackets', 0)}

 **Compliance Rates:**
- Helmet Compliance: {analysis.get('helmet_compliance_rate', 0):.1f}%
- Jacket Compliance: {analysis.get('jacket_compliance_rate', 0):.1f}%
- Overall Compliance: {analysis.get('overall_compliance_rate', 0):.1f}%

 **Violations:**
{chr(10).join(f"- {v}" for v in analysis.get('violations', [])) if analysis.get('violations') else "- None"}

 **Safety Status:** {'COMPLIANT' if analysis.get('is_compliant', False) else 'VIOLATION DETECTED'}

⏱ **Processing Time:** {results.get('processing_time', 0):.3f} seconds
 **Device:** {results.get('device', 'Unknown')}
"""

 # Create statistics JSON
 stats = {
 'timestamp': datetime.now().isoformat(),
 'total_persons': analysis.get('total_persons', 0),
 'helmet_compliance': analysis.get('helmet_compliance_rate', 0),
 'jacket_compliance': analysis.get('jacket_compliance_rate', 0),
 'violations': len(analysis.get('violations', [])),
 'is_compliant': analysis.get('is_compliant', False),
 'processing_time': results.get('processing_time', 0)
 }

 return annotated_image, analysis_text, json.dumps(stats, indent=2)

 except Exception as e:
 return image, f"Processing error: {str(e)}", "{}"

def process_video_file(video_path: str, progress=gr.Progress()) -> Tuple[str, str]:
 """
 Process uploaded video file

 Args:
 video_path: Path to uploaded video file
 progress: Gradio progress tracker

 Returns:
 Tuple of (analysis_text, statistics_json)
 """
 if not video_path:
 return "No video file provided", "{}"

 detector = initialize_detector()

 try:
 cap = cv2.VideoCapture(video_path)
 if not cap.isOpened():
 return "Error: Could not open video file", "{}"

 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 fps = cap.get(cv2.CAP_PROP_FPS)

 frame_results = []
 frame_count = 0
 processed_frames = 0

 # Process every 30th frame for efficiency
 frame_skip = 30

 progress(0, desc="Starting video analysis...")

 while True:
 ret, frame = cap.read()
 if not ret:
 break

 frame_count += 1

 if frame_count % frame_skip == 0:
 processed_frames += 1

 # Update progress
 progress_pct = (frame_count / total_frames)
 progress(progress_pct, desc=f"Processing frame {frame_count}/{total_frames}")

 # Run detection
 results = detector.detect_safety_equipment(frame)

 if 'error' not in results:
 analysis = results.get('safety_analysis', {})

 frame_data = {
 'frame': frame_count,
 'timestamp': frame_count / fps if fps > 0 else 0,
 'persons': analysis.get('total_persons', 0),
 'helmet_compliance': analysis.get('helmet_compliance_rate', 0),
 'jacket_compliance': analysis.get('jacket_compliance_rate', 0),
 'violations': len(analysis.get('violations', [])),
 'is_compliant': analysis.get('is_compliant', True)
 }

 frame_results.append(frame_data)

 cap.release()

 # Calculate overall statistics
 if frame_results:
 total_persons = sum(f['persons'] for f in frame_results)
 violation_frames = sum(1 for f in frame_results if not f['is_compliant'])
 avg_helmet_compliance = np.mean([f['helmet_compliance'] for f in frame_results])
 avg_jacket_compliance = np.mean([f['jacket_compliance'] for f in frame_results])

 analysis_text = f"""
 **Video Analysis Complete**

 **Video Information:**
- Total Frames: {total_frames:,}
- Analyzed Frames: {processed_frames:,}
- Duration: {total_frames/fps:.1f} seconds
- FPS: {fps:.1f}

 **Detection Summary:**
- Total Persons Detected: {total_persons}
- Frames with Violations: {violation_frames}/{processed_frames}
- Violation Rate: {(violation_frames/processed_frames*100):.1f}%

 **Average Compliance:**
- Helmet Compliance: {avg_helmet_compliance:.1f}%
- Jacket Compliance: {avg_jacket_compliance:.1f}%

 **Overall Status:** {'MOSTLY COMPLIANT' if violation_frames/processed_frames < 0.1 else 'VIOLATIONS DETECTED'}
"""

 stats = {
 'video_info': {
 'total_frames': total_frames,
 'analyzed_frames': processed_frames,
 'duration_seconds': total_frames/fps if fps > 0 else 0,
 'fps': fps
 },
 'detection_summary': {
 'total_persons': total_persons,
 'violation_frames': violation_frames,
 'violation_rate': violation_frames/processed_frames*100 if processed_frames > 0 else 0,
 'avg_helmet_compliance': avg_helmet_compliance,
 'avg_jacket_compliance': avg_jacket_compliance
 },
 'frame_data': frame_results[:100] # Limit to first 100 frames for JSON size
 }

 return analysis_text, json.dumps(stats, indent=2)
 else:
 return "No frames could be processed", "{}"

 except Exception as e:
 return f"Video processing error: {str(e)}", "{}"

def detect_available_cameras():
 """Detect available cameras and return as dropdown options"""
 global available_cameras, camera_manager

 try:
 cameras = camera_manager.detect_cameras()
 available_cameras = cameras

 # Create dropdown options
 camera_options = []
 for cam in cameras:
 label = f"{cam['name']} ({cam['type']}) - {cam['width']}x{cam['height']}"
 camera_options.append((label, cam['source']))

 if not camera_options:
 camera_options = [("No cameras detected", -1)]
 return gr.update(choices=camera_options, value=-1)

 # Use the first available camera as default
 default_value = camera_options[0][1]
 return gr.update(choices=camera_options, value=default_value)

 except Exception as e:
 return gr.update(choices=[(f"Error detecting cameras: {str(e)}", -1)], value=-1)

def update_camera_source_placeholder(camera_type):
 """Update camera source placeholder and info based on camera type"""
 examples = {
 "webcam": {
 "placeholder": "0",
 "info": "Enter camera index: 0 (default camera), 1 (external camera), etc."
 },
 "ip_camera": {
 "placeholder": "192.168.1.100:8080",
 "info": "Enter IP address:port or full HTTP URL of the IP camera"
 },
 "rtsp": {
 "placeholder": "rtsp://192.168.1.100:554/stream",
 "info": "Enter RTSP URL (rtsp://ip:port/path) or just IP address for auto URL"
 },
 "file": {
 "placeholder": "/path/to/video.mp4",
 "info": "Enter full path to video file (MP4, AVI, MOV, etc.)"
 }
 }

 selected = examples.get(camera_type, examples["ip_camera"])
 return gr.update(
 placeholder=selected["placeholder"],
 info=selected["info"]
 )

def add_custom_camera(camera_type, source_input, username, password):
 """Add custom camera source (IP camera, RTSP, etc.)"""
 global available_cameras

 try:
 # Validate input
 if not source_input or not source_input.strip():
 return " Error: Camera source cannot be empty", gr.update()

 # Process source based on type
 if camera_type == "webcam":
 try:
 source = int(source_input.strip())
 if source < 0:
 return " Error: Webcam source must be a non-negative number (0, 1, 2, etc.)", gr.update()
 except ValueError:
 return " Error: Webcam source must be a valid number (0, 1, 2, etc.)", gr.update()
 else:
 source = source_input.strip()
 if not source:
 return " Error: Camera source cannot be empty", gr.update()

 # Create camera configuration
 try:
 config = camera_manager.create_camera_source(
 camera_type=camera_type,
 source=source,
 username=username.strip() if username and username.strip() else None,
 password=password.strip() if password and password.strip() else None
 )
 except Exception as config_error:
 return f" Configuration error: {str(config_error)}", gr.update()

 # Validate camera source
 try:
 is_valid, message = camera_manager.validate_camera_source(config)
 except Exception as validation_error:
 return f" Validation error: {str(validation_error)}", gr.update()

 if is_valid:
 # Add to available cameras
 camera_info = {
 'name': f"Custom {camera_type.replace('_', ' ').title()}",
 'type': camera_type,
 'source': source,
 'status': 'available'
 }
 available_cameras.append(camera_info)

 # Update dropdown
 camera_options = []
 for cam in available_cameras:
 label = f"{cam['name']} ({cam['type']})"
 camera_options.append((label, cam['source']))

 return f" Camera added successfully: {message}", gr.update(choices=camera_options)
 else:
 return f" Camera validation failed: {message}", gr.update()

 except Exception as e:
 return f" Unexpected error: {str(e)}", gr.update()

def start_live_camera(selected_camera_source):
 """Start live camera feed with selected camera"""
 global camera_active, live_camera_thread, current_camera_source

 if camera_active:
 return "🟡 Camera is already active"

 if selected_camera_source in [-1, "none", None] or not str(selected_camera_source).strip():
 return " No valid camera selected. Please detect cameras first and select one."

 current_camera_source = selected_camera_source

 def camera_loop():
 global camera_active, detection_history, violation_stats, current_frame, current_annotated_frame

 # Try to open the selected camera
 cap = cv2.VideoCapture(current_camera_source)

 # Configure camera for optimal performance
 cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce buffer to minimize delay
 cap.set(cv2.CAP_PROP_FPS, 30) # Set desired FPS

 if not cap.isOpened():
 return

 camera_active = True
 detector = initialize_detector()

 while camera_active:
 ret, frame = cap.read()
 if not ret:
 break

 # Store current frame for display
 current_frame = frame.copy()

 try:
 # Run detection
 results = detector.detect_safety_equipment(frame)

 if 'error' not in results:
 # Draw detections on frame
 annotated_frame = detector.draw_detections(frame, results)
 current_annotated_frame = annotated_frame.copy()

 analysis = results.get('safety_analysis', {})

 # Update statistics
 if not analysis.get('is_compliant', True):
 violation_stats["total"] += 1
 violations = analysis.get('violations', [])
 for v in violations:
 if 'helmet' in v.lower():
 violation_stats["helmet"] += 1
 if 'jacket' in v.lower():
 violation_stats["jacket"] += 1

 # Store detection history (keep last 100)
 detection_data = {
 'timestamp': datetime.now(),
 'persons': analysis.get('total_persons', 0),
 'helmet_compliance': analysis.get('helmet_compliance_rate', 0),
 'jacket_compliance': analysis.get('jacket_compliance_rate', 0),
 'violations': len(analysis.get('violations', [])),
 'is_compliant': analysis.get('is_compliant', True)
 }

 detection_history.append(detection_data)
 if len(detection_history) > 100:
 detection_history.pop(0)
 else:
 # If detection failed, still store the frame
 current_annotated_frame = frame.copy()

 except Exception as e:
 print(f"Detection error: {e}")
 current_annotated_frame = frame.copy()

 # Small delay to prevent excessive CPU usage while maintaining high FPS
 time.sleep(0.01) # 10ms delay allows up to ~100 FPS

 cap.release()

 live_camera_thread = threading.Thread(target=camera_loop, daemon=True)
 live_camera_thread.start()

 return " Live camera started successfully"

def stop_live_camera():
 """Stop live camera feed"""
 global camera_active
 camera_active = False
 return " Live camera stopped"

def get_live_frame() -> Optional[np.ndarray]:
 """Get current live camera frame with detections"""
 global current_annotated_frame

 if current_annotated_frame is not None:
 # Convert BGR (OpenCV format) to RGB (Gradio web interface format)
 return cv2.cvtColor(current_annotated_frame, cv2.COLOR_BGR2RGB)
 return None

def reset_person_tracking():
 """Reset person tracking and statistics"""
 global current_detector, detection_history, violation_stats

 # Reset person tracker
 if current_detector and hasattr(current_detector, 'person_tracker'):
 current_detector.person_tracker.reset()

 # Reset statistics
 detection_history.clear()
 violation_stats = {"total": 0, "helmet": 0, "jacket": 0}

 return " Person tracking and statistics reset successfully"

def get_live_stats() -> Tuple[str, str]:
 """Get live camera statistics with tracking information"""
 global detection_history, violation_stats, current_detector

 if not detection_history:
 return "No live data available yet", "{}"

 recent_data = detection_history[-10:] # Last 10 detections

 total_persons = sum(d['persons'] for d in recent_data)
 avg_helmet = np.mean([d['helmet_compliance'] for d in recent_data])
 avg_jacket = np.mean([d['jacket_compliance'] for d in recent_data])
 recent_violations = sum(d['violations'] for d in recent_data)

 # Get tracking information if available
 tracking_info = ""
 if current_detector and hasattr(current_detector, 'person_tracker'):
 tracking_summary = current_detector.person_tracker.get_tracking_summary()
 tracking_info = f"""

 **Person Tracking:**
- Unique Persons Tracked: {tracking_summary.get('total_tracked_persons', 0)}
- Currently Active: {tracking_summary.get('active_persons', 0)}
- Next Person ID: {tracking_summary.get('next_person_id', 0)}
- Compliance Rate: {tracking_summary.get('compliance_rate', 0):.1f}%
"""

 stats_text = f"""
 **Live Camera Statistics**

 **Recent Activity (Last 10 detections):**
- Total Person Detections: {total_persons}
- Average Helmet Compliance: {avg_helmet:.1f}%
- Average Jacket Compliance: {avg_jacket:.1f}%
- Recent Violations: {recent_violations}
{tracking_info}
 **Session Totals:**
- Total Violations: {violation_stats['total']}
- Helmet Violations: {violation_stats['helmet']}
- Jacket Violations: {violation_stats['jacket']}

 **Status:** {'🟢 Active' if camera_active else ' Inactive'}
 **Data Points:** {len(detection_history)}
"""

 stats = {
 'recent_activity': {
 'total_persons': total_persons,
 'avg_helmet_compliance': avg_helmet,
 'avg_jacket_compliance': avg_jacket,
 'recent_violations': recent_violations
 },
 'session_totals': violation_stats.copy(),
 'camera_active': camera_active,
 'data_points': len(detection_history),
 'latest_detection': detection_history[-1] if detection_history else None,
 'tracking_info': current_detector.person_tracker.get_tracking_summary() if current_detector and hasattr(current_detector, 'person_tracker') else None
 }

 return stats_text, json.dumps(stats, indent=2, default=str)

def create_compliance_chart() -> Optional[gr.Plot]:
 """Create compliance rate chart from live data"""
 global detection_history

 if len(detection_history) < 2:
 return None

 # Prepare data for plotting
 timestamps = [d['timestamp'] for d in detection_history[-50:]] # Last 50 points
 helmet_rates = [d['helmet_compliance'] for d in detection_history[-50:]]
 jacket_rates = [d['jacket_compliance'] for d in detection_history[-50:]]

 # Create plot
 fig = go.Figure()

 fig.add_trace(go.Scatter(
 x=timestamps,
 y=helmet_rates,
 mode='lines+markers',
 name='Helmet Compliance',
 line=dict(color='blue')
 ))

 fig.add_trace(go.Scatter(
 x=timestamps,
 y=jacket_rates,
 mode='lines+markers',
 name='Jacket Compliance',
 line=dict(color='orange')
 ))

 fig.update_layout(
 title='Live Safety Compliance Rates',
 xaxis_title='Time',
 yaxis_title='Compliance Rate (%)',
 yaxis=dict(range=[0, 100]),
 template='plotly_white'
 )

 return fig

# Create Gradio interface
def create_interface():
 """Create the main Gradio interface"""

 with gr.Blocks(
 title=" Thermal Power Plant Safety Detection System",
 theme=gr.themes.Soft(),
 css="""
 .gradio-container {
 max-width: 1200px;
 margin: auto;
 }
 .status-compliant {
 background-color: #d4edda;
 border-color: #c3e6cb;
 color: #155724;
 }
 .status-violation {
 background-color: #f8d7da;
 border-color: #f5c6cb;
 color: #721c24;
 }
 """
 ) as app:

 gr.Markdown("""
 # Thermal Power Plant Safety Detection System

 Advanced AI-powered safety monitoring system for detecting safety equipment compliance.

 **Features:**
 - Live camera monitoring
 - Image analysis
 - Video file processing
 - Real-time statistics
 - Violation detection
 """)

 with gr.Tabs():
 # Tab 1: Live Camera Feed
 with gr.Tab(" Live Camera", id="live_camera"):
 gr.Markdown("## Live Safety Monitoring")
 gr.Markdown("Monitor safety compliance in real-time using your camera or CCTV.")

 # Camera Selection Section
 with gr.Group():
 gr.Markdown("### Camera Selection")

 with gr.Row():
 with gr.Column():
 detect_cameras_btn = gr.Button(" Detect Cameras", variant="secondary")
 camera_dropdown = gr.Dropdown(
 label="Select Camera",
 choices=[("Click 'Detect Cameras' first", "none")],
 value="none",
 interactive=True
 )

 with gr.Column():
 camera_status = gr.Textbox(
 label="Camera Status",
 value="Ready to detect cameras",
 interactive=False,
 lines=2
 )

 # Custom Camera Section
 with gr.Group():
 gr.Markdown("### Add Custom Camera (IP/RTSP/CCTV)")

 with gr.Row():
 camera_type = gr.Dropdown(
 label="Camera Type",
 choices=[
 ("Webcam (USB/Built-in)", "webcam"),
 ("IP Camera (HTTP)", "ip_camera"),
 ("RTSP Stream", "rtsp"),
 ("Video File", "file")
 ],
 value="ip_camera"
 )

 camera_source = gr.Textbox(
 label="Camera Source",
 placeholder="Enter camera source (examples will update based on type)",
 lines=1,
 info="Examples: 0 (webcam), 192.168.1.100 (IP), rtsp://192.168.1.100/stream (RTSP)"
 )

 with gr.Row():
 camera_username = gr.Textbox(
 label="Username (if required)",
 placeholder="admin",
 lines=1
 )

 camera_password = gr.Textbox(
 label="Password (if required)",
 placeholder="password",
 type="password",
 lines=1
 )

 add_camera_btn = gr.Button(" Add Camera", variant="secondary")

 # Control Section
 with gr.Group():
 gr.Markdown("### Camera Control")

 with gr.Row():
 with gr.Column():
 start_btn = gr.Button("🟢 Start Camera", variant="primary")
 stop_btn = gr.Button(" Stop Camera", variant="secondary")

 with gr.Column():
 refresh_stats_btn = gr.Button(" Refresh Stats", variant="secondary")
 reset_tracking_btn = gr.Button(" Reset Person Tracking", variant="secondary")

 with gr.Row():
 with gr.Column():
 live_camera_image = gr.Image(
 label=" Live Camera Feed",
 height=600,
 width=800,
 interactive=False
 )

 with gr.Column():
 live_stats_text = gr.Textbox(
 label=" Live Statistics",
 lines=15,
 max_lines=20,
 interactive=False
 )

 live_stats_json = gr.JSON(
 label=" Detailed Stats (JSON)",
 visible=False
 )

 # compliance_chart = gr.Plot(label=" Compliance Trends")

 # Event handlers for camera detection and selection
 detect_cameras_btn.click(
 detect_available_cameras,
 outputs=camera_dropdown
 )

 # Update camera source placeholder when camera type changes
 camera_type.change(
 update_camera_source_placeholder,
 inputs=camera_type,
 outputs=camera_source
 )

 add_camera_btn.click(
 add_custom_camera,
 inputs=[camera_type, camera_source, camera_username, camera_password],
 outputs=[camera_status, camera_dropdown]
 )

 # Event handlers for camera control
 start_btn.click(
 start_live_camera,
 inputs=camera_dropdown,
 outputs=camera_status
 )

 stop_btn.click(
 stop_live_camera,
 outputs=camera_status
 )

 reset_tracking_btn.click(
 reset_person_tracking,
 outputs=camera_status
 )

 refresh_stats_btn.click(
 get_live_stats,
 outputs=[live_stats_text, live_stats_json]
 )

 # Auto-refresh live camera image and stats
 def refresh_live_image():
 return get_live_frame()

 def refresh_all():
 """Refresh both live image and stats"""
 stats_text, stats_json = get_live_stats()
 live_image = get_live_frame()
 return live_image, stats_text, stats_json

 # Set up periodic refresh for live camera image and stats
 refresh_stats_btn.click(refresh_all, outputs=[live_camera_image, live_stats_text, live_stats_json])

 # Add automatic refresh using a timer component
 timer = gr.Timer(value=2.0) # Refresh every 2 seconds
 timer.tick(refresh_all, outputs=[live_camera_image, live_stats_text, live_stats_json])

 # Tab 2: Image Analysis
 with gr.Tab(" Image Analysis", id="image_analysis"):
 gr.Markdown("## Safety Equipment Detection in Images")
 gr.Markdown("Upload an image to analyze safety compliance.")

 with gr.Row():
 with gr.Column():
 input_image = gr.Image(
 label=" Upload Image",
 type="numpy",
 height=400
 )

 analyze_btn = gr.Button(" Analyze Image", variant="primary")

 with gr.Column():
 output_image = gr.Image(
 label=" Detection Results",
 height=400
 )

 with gr.Row():
 with gr.Column():
 analysis_text = gr.Textbox(
 label=" Analysis Report",
 lines=15,
 max_lines=20,
 interactive=False
 )

 with gr.Column():
 analysis_json = gr.JSON(
 label=" Detailed Results (JSON)",
 visible=False
 )

 # Event handlers
 analyze_btn.click(
 process_image,
 inputs=input_image,
 outputs=[output_image, analysis_text, analysis_json]
 )

 # Auto-analyze on image upload
 input_image.change(
 process_image,
 inputs=input_image,
 outputs=[output_image, analysis_text, analysis_json]
 )

 # Tab 3: Video Analysis
 with gr.Tab(" Video Analysis", id="video_analysis"):
 gr.Markdown("## Safety Compliance Video Analysis")
 gr.Markdown("Upload a video file to analyze safety compliance over time.")

 with gr.Row():
 with gr.Column():
 input_video = gr.Video(
 label=" Upload Video File",
 height=300
 )

 analyze_video_btn = gr.Button(" Analyze Video", variant="primary")

 with gr.Column():
 video_progress = gr.Textbox(
 label="⏳ Processing Status",
 interactive=False,
 lines=3
 )

 with gr.Row():
 with gr.Column():
 video_analysis_text = gr.Textbox(
 label=" Video Analysis Report",
 lines=15,
 max_lines=20,
 interactive=False
 )

 with gr.Column():
 video_analysis_json = gr.JSON(
 label=" Detailed Results (JSON)",
 visible=False
 )

 # Event handlers
 analyze_video_btn.click(
 process_video_file,
 inputs=input_video,
 outputs=[video_analysis_text, video_analysis_json]
 )

 # Tab 4: System Information
 with gr.Tab("ℹ System Info", id="system_info"):
 gr.Markdown("## System Configuration and Status")

 with gr.Row():
 with gr.Column():
 # System status
 detector = initialize_detector()

 system_info = f"""
** System Information:**
- Device: {detector.device.upper()}
- Model: {config.model.name}
- Confidence Threshold: {config.model.confidence_threshold}
- IoU Threshold: {config.model.iou_threshold}

** Configuration:**
- Environment: {config.environment.value}
- Max Workers: {config.performance.max_workers}
- GPU Acceleration: {config.performance.enable_gpu_acceleration}

** Security:**
- API Authentication: {config.security.enable_api_key_auth}
- Rate Limiting: {config.security.rate_limit_requests} req/min
- Max File Size: {config.security.max_file_size_mb}MB

** Safety Classes:**
- Helmets: {', '.join(config.raw_config.get('classes', {}).get('helmet', ['helmet']))}
- Jackets: {', '.join(config.raw_config.get('classes', {}).get('reflective_jacket', ['jacket']))}
- Persons: {', '.join(config.raw_config.get('classes', {}).get('person', ['person']))}
"""

 gr.Markdown(system_info)

 with gr.Column():
 gr.Markdown("""
** Quick Start Guide:**

1. **Live Monitoring**: Use the "Live Camera" tab to start real-time monitoring
2. **Image Analysis**: Upload images in the "Image Analysis" tab
3. **Video Processing**: Analyze video files in the "Video Analysis" tab

** Safety Requirements:**
- Safety helmets/hard hats required
- High-visibility reflective jackets required
- Violations are automatically detected and reported

** Troubleshooting:**
- Ensure camera permissions are granted
- Check that your camera is not being used by other applications
- For video analysis, supported formats: MP4, AVI, MOV

** Performance Tips:**
- Use good lighting for better detection accuracy
- Ensure people are clearly visible in the frame
- For live monitoring, position camera to capture work areas
""")

 gr.Markdown("""
---

** Safety Detection System v2.0** | Built with for thermal power plant safety

*This system is designed to assist with safety monitoring but should not be the sole method of ensuring workplace safety.*
""")

 return app

def main():
 """Main function to run the Gradio app"""
 print(" Starting Safety Detection System Web Interface...")
 print(" Initializing components...")

 # Initialize detector to check system
 try:
 detector = initialize_detector()
 print(f" Detector initialized on device: {detector.device}")
 except Exception as e:
 print(f" Error initializing detector: {e}")
 return

 # Create and launch interface
 app = create_interface()

 print(" Launching web interface...")
 print(" Open your browser and go to the URL shown below")

 # Launch with camera access enabled
 app.launch(
 server_name="0.0.0.0", # Allow external access
 server_port=7860, # Gradio default port
 share=False, # Set to True for public sharing
 debug=False,
 show_error=True,
 quiet=False
 )

if __name__ == "__main__":
 main()
