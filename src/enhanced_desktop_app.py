#!/usr/bin/env python3
"""
Enhanced Desktop Safety Detection Application with Camera Selection

This enhanced desktop application provides:
- Interactive camera selection
- Multiple camera source support (webcam, IP camera, RTSP, video files)
- Real-time safety monitoring
- Configuration management
- Enhanced UI with camera controls
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime
import json
import os
from typing import Optional, Dict, List, Tuple

try:
    from .safety_detector import SafetyDetector
    from .camera_manager import CameraManager
except ImportError:
    from safety_detector import SafetyDetector
    from camera_manager import CameraManager


class EnhancedDesktopApp:
    """Enhanced desktop application with comprehensive camera management"""
    
    def __init__(self):
        """Initialize the enhanced desktop application"""
        self.root = tk.Tk()
        self.root.title("🦺 Safety Detection System - Enhanced Desktop")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.detector = None
        self.camera_manager = CameraManager()
        
        # Camera control variables
        self.camera_active = False
        self.current_camera_source = None
        self.current_frame = None
        self.current_annotated_frame = None
        self.detection_stats = {
            'total_frames': 0,
            'violation_frames': 0,
            'current_persons': 0,
            'helmet_compliance': 0.0,
            'jacket_compliance': 0.0
        }
        
        # Available cameras list
        self.available_cameras = []
        
        # Threading
        self.camera_thread = None
        self.stats_update_thread = None
        
        # Create UI
        self.create_ui()
        
        # Initialize detector
        self.initialize_detector()
        
        # Start periodic UI updates
        self.start_ui_updates()
    
    def create_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🦺 Enhanced Safety Detection System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Camera selection frame
        self.create_camera_selection_frame(main_frame)
        
        # Control buttons frame
        self.create_control_frame(main_frame)
        
        # Main display area
        self.create_display_area(main_frame)
        
        # Statistics panel
        self.create_statistics_panel(main_frame)
    
    def create_camera_selection_frame(self, parent):
        """Create camera selection interface"""
        camera_frame = ttk.LabelFrame(parent, text="📷 Camera Selection", padding="10")
        camera_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        camera_frame.columnconfigure(1, weight=1)
        
        # Camera detection
        ttk.Button(camera_frame, text="🔍 Detect Cameras", 
                  command=self.detect_cameras).grid(row=0, column=0, padx=(0, 10))
        
        # Camera dropdown
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(camera_frame, textvariable=self.camera_var, 
                                           state="readonly", width=50)
        self.camera_dropdown.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Test camera button
        ttk.Button(camera_frame, text="🧪 Test Camera", 
                  command=self.test_selected_camera).grid(row=0, column=2)
        
        # Custom camera section
        ttk.Separator(camera_frame, orient='horizontal').grid(row=1, column=0, columnspan=3, 
                                                             sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(camera_frame, text="Add Custom Camera:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        
        # Camera type selection
        self.camera_type_var = tk.StringVar(value="webcam")
        camera_type_frame = ttk.Frame(camera_frame)
        camera_type_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(camera_type_frame, text="Webcam", variable=self.camera_type_var, 
                       value="webcam", command=self.update_source_placeholder).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(camera_type_frame, text="IP Camera", variable=self.camera_type_var, 
                       value="ip_camera", command=self.update_source_placeholder).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(camera_type_frame, text="RTSP Stream", variable=self.camera_type_var, 
                       value="rtsp", command=self.update_source_placeholder).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(camera_type_frame, text="Video File", variable=self.camera_type_var, 
                       value="file", command=self.update_source_placeholder).pack(side=tk.LEFT)
        
        # Camera source input
        source_frame = ttk.Frame(camera_frame)
        source_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        source_frame.columnconfigure(0, weight=1)
        
        self.source_var = tk.StringVar()
        self.source_entry = ttk.Entry(source_frame, textvariable=self.source_var, width=40)
        self.source_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        ttk.Button(source_frame, text="📁 Browse", 
                  command=self.browse_video_file).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(source_frame, text="➕ Add Camera", 
                  command=self.add_custom_camera).grid(row=0, column=2)
        
        # Source hint label
        self.source_hint_var = tk.StringVar(value="Enter camera index (0, 1, 2...)")
        self.source_hint_label = ttk.Label(camera_frame, textvariable=self.source_hint_var, 
                                          font=("Arial", 9), foreground="gray")
        self.source_hint_label.grid(row=5, column=0, columnspan=3, sticky=tk.W)
        
        # Authentication for IP cameras
        auth_frame = ttk.Frame(camera_frame)
        auth_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(auth_frame, text="Username:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.username_var = tk.StringVar()
        ttk.Entry(auth_frame, textvariable=self.username_var, width=15).grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(auth_frame, text="Password:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.password_var = tk.StringVar()
        ttk.Entry(auth_frame, textvariable=self.password_var, show="*", width=15).grid(row=0, column=3)
    
    def create_control_frame(self, parent):
        """Create control buttons frame"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Camera controls
        ttk.Button(control_frame, text="🟢 Start Camera", 
                  command=self.start_camera).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(control_frame, text="🔴 Stop Camera", 
                  command=self.stop_camera).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Recording controls
        self.recording_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="📹 Record Video", 
                       variable=self.recording_var).pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Button(control_frame, text="📁 Choose Save Location", 
                  command=self.choose_save_location).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Statistics controls
        ttk.Button(control_frame, text="🔄 Reset Statistics", 
                  command=self.reset_statistics).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(control_frame, text="💾 Save Report", 
                  command=self.save_detection_report).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Settings
        ttk.Button(control_frame, text="⚙️ Settings", 
                  command=self.open_settings).pack(fill=tk.X, pady=(0, 5))
    
    def create_display_area(self, parent):
        """Create video display area"""
        display_frame = ttk.LabelFrame(parent, text="📹 Live Video Feed", padding="5")
        display_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Video canvas
        self.video_canvas = tk.Canvas(display_frame, bg="black", width=640, height=480)
        self.video_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready - Select a camera to start")
        self.status_label = ttk.Label(display_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, pady=(5, 0))
    
    def create_statistics_panel(self, parent):
        """Create statistics display panel"""
        stats_frame = ttk.LabelFrame(parent, text="📊 Detection Statistics", padding="10")
        stats_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Statistics display
        stats_text_frame = ttk.Frame(stats_frame)
        stats_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(stats_text_frame, height=10, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(stats_text_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def initialize_detector(self):
        """Initialize the safety detector"""
        try:
            self.detector = SafetyDetector(confidence_threshold=0.5)
            self.log_message("✅ Safety detector initialized successfully")
        except Exception as e:
            self.log_message(f"❌ Error initializing detector: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize detector: {e}")
    
    def detect_cameras(self):
        """Detect available cameras"""
        try:
            self.log_message("🔍 Detecting cameras...")
            cameras = self.camera_manager.detect_cameras()
            self.available_cameras = cameras
            
            # Update dropdown
            camera_options = []
            for cam in cameras:
                label = f"{cam['name']} ({cam['type']}) - {cam.get('width', 'N/A')}x{cam.get('height', 'N/A')}"
                camera_options.append(label)
            
            if camera_options:
                self.camera_dropdown['values'] = camera_options
                self.camera_dropdown.current(0)
                self.log_message(f"✅ Found {len(cameras)} camera(s)")
            else:
                self.camera_dropdown['values'] = ["No cameras detected"]
                self.log_message("⚠️ No cameras detected")
                
        except Exception as e:
            self.log_message(f"❌ Error detecting cameras: {e}")
            messagebox.showerror("Detection Error", f"Failed to detect cameras: {e}")
    
    def test_selected_camera(self):
        """Test the selected camera"""
        try:
            selected_index = self.camera_dropdown.current()
            if selected_index < 0 or selected_index >= len(self.available_cameras):
                messagebox.showwarning("Selection Error", "Please select a camera first")
                return
            
            camera = self.available_cameras[selected_index]
            self.log_message(f"🧪 Testing camera: {camera['name']}")
            
            success, properties = self.camera_manager.test_camera_connection(camera['source'])
            
            if success:
                self.log_message(f"✅ Camera test successful: {properties}")
                messagebox.showinfo("Camera Test", f"Camera test successful!\n\nProperties:\n{json.dumps(properties, indent=2)}")
            else:
                self.log_message(f"❌ Camera test failed: {properties}")
                messagebox.showerror("Camera Test", f"Camera test failed:\n{properties}")
                
        except Exception as e:
            self.log_message(f"❌ Error testing camera: {e}")
            messagebox.showerror("Test Error", f"Failed to test camera: {e}")
    
    def update_source_placeholder(self):
        """Update source entry placeholder based on camera type"""
        camera_type = self.camera_type_var.get()
        
        placeholders = {
            'webcam': "0",
            'ip_camera': "192.168.1.100:8080",
            'rtsp': "rtsp://192.168.1.100:554/stream",
            'file': "/path/to/video.mp4"
        }
        
        hints = {
            'webcam': "Enter camera index (0, 1, 2...)",
            'ip_camera': "Enter IP address:port or full HTTP URL",
            'rtsp': "Enter RTSP URL or IP address",
            'file': "Enter path to video file or browse"
        }
        
        self.source_var.set(placeholders.get(camera_type, ""))
        self.source_hint_var.set(hints.get(camera_type, ""))
    
    def browse_video_file(self):
        """Browse for video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.source_var.set(file_path)
    
    def add_custom_camera(self):
        """Add custom camera source"""
        try:
            camera_type = self.camera_type_var.get()
            source = self.source_var.get().strip()
            username = self.username_var.get().strip() or None
            password = self.password_var.get().strip() or None
            
            if not source:
                messagebox.showwarning("Input Error", "Please enter a camera source")
                return
            
            # Create camera configuration
            config = self.camera_manager.create_camera_source(
                camera_type=camera_type,
                source=source,
                username=username,
                password=password
            )
            
            # Validate camera
            is_valid, message = self.camera_manager.validate_camera_source(config)
            
            if is_valid:
                # Add to available cameras
                camera_info = {
                    'name': f"Custom {camera_type.replace('_', ' ').title()}",
                    'type': camera_type,
                    'source': source,
                    'status': 'available'
                }
                self.available_cameras.append(camera_info)
                
                # Update dropdown
                current_values = list(self.camera_dropdown['values'])
                new_label = f"{camera_info['name']} ({camera_info['type']})"
                current_values.append(new_label)
                self.camera_dropdown['values'] = current_values
                self.camera_dropdown.set(new_label)
                
                self.log_message(f"✅ Camera added successfully: {message}")
                messagebox.showinfo("Success", f"Camera added successfully!\n{message}")
            else:
                self.log_message(f"❌ Camera validation failed: {message}")
                messagebox.showerror("Validation Error", f"Camera validation failed:\n{message}")
                
        except Exception as e:
            self.log_message(f"❌ Error adding camera: {e}")
            messagebox.showerror("Add Camera Error", f"Failed to add camera: {e}")
    
    def start_camera(self):
        """Start camera feed"""
        if self.camera_active:
            messagebox.showinfo("Info", "Camera is already active")
            return
        
        try:
            selected_index = self.camera_dropdown.current()
            if selected_index < 0 or selected_index >= len(self.available_cameras):
                messagebox.showwarning("Selection Error", "Please select a camera first")
                return
            
            camera = self.available_cameras[selected_index]
            self.current_camera_source = camera['source']
            
            self.camera_active = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.log_message(f"🟢 Camera started: {camera['name']}")
            self.status_var.set("Camera active - Processing live feed")
            
        except Exception as e:
            self.log_message(f"❌ Error starting camera: {e}")
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_active = False
        self.current_camera_source = None
        self.log_message("🔴 Camera stopped")
        self.status_var.set("Camera stopped")
    
    def camera_loop(self):
        """Main camera processing loop"""
        cap = None
        writer = None
        
        try:
            cap = cv2.VideoCapture(self.current_camera_source)
            
            if not cap.isOpened():
                self.log_message(f"❌ Failed to open camera: {self.current_camera_source}")
                return
            
            # Configure camera
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Setup video writer if recording
            if self.recording_var.get():
                save_path = getattr(self, 'save_path', 'safety_detection_output.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                self.log_message(f"📹 Recording to: {save_path}")
            
            while self.camera_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.current_frame = frame.copy()
                
                # Run detection
                if self.detector:
                    results = self.detector.detect_safety_equipment(frame)
                    annotated_frame = self.detector.draw_detections(frame, results)
                    self.current_annotated_frame = annotated_frame.copy()
                    
                    # Update statistics
                    self.update_detection_stats(results)
                    
                    # Save frame if recording
                    if writer:
                        writer.write(annotated_frame)
                else:
                    self.current_annotated_frame = frame.copy()
                
                time.sleep(0.01)  # Limit CPU usage
                
        except Exception as e:
            self.log_message(f"❌ Camera loop error: {e}")
        finally:
            if cap:
                cap.release()
            if writer:
                writer.release()
    
    def update_detection_stats(self, results):
        """Update detection statistics"""
        self.detection_stats['total_frames'] += 1
        
        if 'safety_analysis' in results:
            analysis = results['safety_analysis']
            self.detection_stats['current_persons'] = analysis.get('total_persons', 0)
            self.detection_stats['helmet_compliance'] = analysis.get('helmet_compliance_rate', 0)
            self.detection_stats['jacket_compliance'] = analysis.get('jacket_compliance_rate', 0)
            
            if not analysis.get('is_compliant', True):
                self.detection_stats['violation_frames'] += 1
    
    def start_ui_updates(self):
        """Start periodic UI updates"""
        self.update_video_display()
        self.update_statistics_display()
        self.root.after(100, self.start_ui_updates)  # Update every 100ms
    
    def update_video_display(self):
        """Update video display canvas"""
        if self.current_annotated_frame is not None:
            try:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(self.current_annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit canvas
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    h, w = frame_rgb.shape[:2]
                    scale = min(canvas_width/w, canvas_height/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    
                    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                    
                    # Convert to PhotoImage
                    from PIL import Image, ImageTk
                    image = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image)
                    
                    # Update canvas
                    self.video_canvas.delete("all")
                    x = (canvas_width - new_w) // 2
                    y = (canvas_height - new_h) // 2
                    self.video_canvas.create_image(x, y, anchor=tk.NW, image=photo)
                    self.video_canvas.image = photo  # Keep a reference
                    
            except Exception as e:
                self.log_message(f"❌ Display update error: {e}")
    
    def update_statistics_display(self):
        """Update statistics text display"""
        stats = self.detection_stats
        total_frames = stats['total_frames']
        violation_frames = stats.get('violation_frames', 0)
        
        violation_rate = (violation_frames / total_frames * 100) if total_frames > 0 else 0
        
        stats_text = f"""🎯 Real-time Detection Statistics

📊 Frame Analysis:
• Total Frames Processed: {total_frames:,}
• Violation Frames: {violation_frames:,}
• Violation Rate: {violation_rate:.1f}%

👥 Current Detection:
• Persons Detected: {stats['current_persons']}
• Helmet Compliance: {stats['helmet_compliance']:.1f}%
• Jacket Compliance: {stats['jacket_compliance']:.1f}%

⏰ Session Info:
• Started: {getattr(self, 'session_start', 'N/A')}
• Camera Status: {'Active' if self.camera_active else 'Inactive'}
• Recording: {'ON' if self.recording_var.get() else 'OFF'}

💡 Controls:
• Press 'q' to quit
• Use control buttons for camera management
• Check settings for detector configuration
"""
        
        # Update text widget
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'total_frames': 0,
            'violation_frames': 0,
            'current_persons': 0,
            'helmet_compliance': 0.0,
            'jacket_compliance': 0.0
        }
        self.log_message("🔄 Statistics reset")
    
    def choose_save_location(self):
        """Choose save location for recording"""
        file_path = filedialog.asksaveasfilename(
            title="Choose Save Location",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if file_path:
            self.save_path = file_path
            self.log_message(f"💾 Save location set: {file_path}")
    
    def save_detection_report(self):
        """Save detection report"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Detection Report",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if file_path:
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'statistics': self.detection_stats,
                    'camera_source': str(self.current_camera_source),
                    'detector_config': {
                        'confidence_threshold': self.detector.confidence_threshold if self.detector else 'N/A',
                        'device': self.detector.device if self.detector else 'N/A'
                    }
                }
                
                with open(file_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                self.log_message(f"💾 Report saved: {file_path}")
                messagebox.showinfo("Success", f"Report saved successfully!\n{file_path}")
                
        except Exception as e:
            self.log_message(f"❌ Error saving report: {e}")
            messagebox.showerror("Save Error", f"Failed to save report: {e}")
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Detector settings
        if self.detector:
            ttk.Label(settings_window, text="Detector Settings", font=("Arial", 12, "bold")).pack(pady=10)
            
            # Confidence threshold
            ttk.Label(settings_window, text="Confidence Threshold:").pack()
            confidence_var = tk.DoubleVar(value=self.detector.confidence_threshold)
            confidence_scale = ttk.Scale(settings_window, from_=0.1, to=1.0, 
                                       variable=confidence_var, orient=tk.HORIZONTAL)
            confidence_scale.pack(fill=tk.X, padx=20)
            
            # Apply button
            def apply_settings():
                self.detector.confidence_threshold = confidence_var.get()
                self.log_message(f"⚙️ Settings updated: confidence={confidence_var.get():.2f}")
                settings_window.destroy()
            
            ttk.Button(settings_window, text="Apply", command=apply_settings).pack(pady=20)
    
    def log_message(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run(self):
        """Run the application"""
        self.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_message("🚀 Enhanced Desktop Safety Detection System started")
        
        # Handle window close
        def on_closing():
            self.stop_camera()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()


if __name__ == "__main__":
    app = EnhancedDesktopApp()
    app.run()
