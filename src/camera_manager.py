#!/usr/bin/env python3
"""
Camera Manager for Safety Detection System

This module provides enhanced camera connectivity options including:
- Webcam detection and management
- IP camera/CCTV connectivity
- RTSP stream support
- Camera configuration and optimization
"""

import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Union
import platform
import subprocess
import re
from urllib.parse import urlparse

class CameraManager:
    """
    Enhanced camera manager for webcam and CCTV connectivity
    """
    
    def __init__(self):
        """Initialize the camera manager"""
        self.active_cameras = {}
        self.camera_configs = {}
        self.system_platform = platform.system().lower()
        
    def detect_cameras(self) -> List[Dict]:
        """
        Detect available cameras on the system
        
        Returns:
            List of camera information dictionaries
        """
        cameras = []
        
        # Test built-in cameras (usually 0-3)
        for camera_id in range(4):
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_info = {
                            'id': camera_id,
                            'name': f'Camera {camera_id}',
                            'type': 'webcam',
                            'source': camera_id,
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'status': 'available'
                        }
                        cameras.append(camera_info)
                
                cap.release()
                
            except Exception as e:
                print(f"Error testing camera {camera_id}: {e}")
                continue
        
        # Add system-specific camera detection
        if self.system_platform == 'darwin':  # macOS
            cameras.extend(self._detect_macos_cameras())
        elif self.system_platform == 'linux':
            cameras.extend(self._detect_linux_cameras())
        elif self.system_platform == 'windows':
            cameras.extend(self._detect_windows_cameras())
        
        return cameras
    
    def _detect_macos_cameras(self) -> List[Dict]:
        """Detect cameras on macOS using system_profiler"""
        cameras = []
        try:
            result = subprocess.run([
                'system_profiler', 'SPCameraDataType', '-xml'
            ], capture_output=True, text=True)
            
            # Parse system profiler output (simplified)
            # In a real implementation, you'd parse the XML properly
            if 'Camera' in result.stdout:
                # This is a simplified detection - would need proper XML parsing
                pass
                
        except Exception as e:
            print(f"Error detecting macOS cameras: {e}")
        
        return cameras
    
    def _detect_linux_cameras(self) -> List[Dict]:
        """Detect cameras on Linux using v4l2"""
        cameras = []
        try:
            # List video devices
            devices = subprocess.run([
                'ls', '/dev/video*'
            ], capture_output=True, text=True)
            
            if devices.returncode == 0:
                for device_path in devices.stdout.strip().split('\n'):
                    if device_path:
                        device_id = int(re.search(r'video(\d+)', device_path).group(1))
                        # Additional info could be gathered using v4l2-ctl
                        pass
                        
        except Exception as e:
            print(f"Error detecting Linux cameras: {e}")
        
        return cameras
    
    def _detect_windows_cameras(self) -> List[Dict]:
        """Detect cameras on Windows using DirectShow"""
        # This would typically use DirectShow or Media Foundation APIs
        # For now, we rely on OpenCV's enumeration
        return []
    
    def create_camera_source(self, 
                           camera_type: str,
                           source: Union[int, str],
                           **kwargs) -> Dict:
        """
        Create a camera source configuration
        
        Args:
            camera_type: Type of camera ('webcam', 'ip_camera', 'rtsp', 'file')
            source: Camera source (index for webcam, URL for IP camera)
            **kwargs: Additional configuration options
            
        Returns:
            Camera source configuration
        """
        config = {
            'type': camera_type,
            'source': source,
            'width': kwargs.get('width', 1280),
            'height': kwargs.get('height', 720),
            'fps': kwargs.get('fps', 30),
            'buffer_size': kwargs.get('buffer_size', 1),
            'timeout': kwargs.get('timeout', 5),
            'retry_attempts': kwargs.get('retry_attempts', 3),
            'retry_delay': kwargs.get('retry_delay', 2)
        }
        
        # Type-specific configurations
        if camera_type == 'ip_camera' or camera_type == 'rtsp':
            config.update({
                'username': kwargs.get('username'),
                'password': kwargs.get('password'),
                'transport': kwargs.get('transport', 'tcp'),  # tcp or udp
                'latency': kwargs.get('latency', 'low')
            })
        
        return config
    
    def validate_camera_source(self, config: Dict) -> Tuple[bool, str]:
        """
        Validate a camera source configuration
        
        Args:
            config: Camera configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Create appropriate source string
            source = self._create_source_string(config)
            
            # Test connection
            cap = cv2.VideoCapture(source)
            
            # Configure camera properties
            self._configure_camera(cap, config)
            
            if not cap.isOpened():
                return False, "Failed to open camera source"
            
            # Try to read a frame
            ret, frame = cap.read()
            if not ret or frame is None:
                return False, "Failed to read frame from camera"
            
            cap.release()
            return True, "Camera source is valid"
            
        except Exception as e:
            return False, f"Error validating camera: {str(e)}"
    
    def _create_source_string(self, config: Dict) -> Union[int, str]:
        """Create OpenCV-compatible source string from config"""
        camera_type = config['type']
        source = config['source']
        
        # Validate source is not empty
        if not source and source != 0:
            raise ValueError("Camera source cannot be empty")
        
        if camera_type == 'webcam':
            try:
                return int(source)
            except (ValueError, TypeError):
                raise ValueError(f"Webcam source must be a valid integer, got: {source}")
        
        elif camera_type == 'ip_camera':
            # Handle various IP camera URL formats
            if isinstance(source, str) and source.strip():
                source = source.strip()
                if source.startswith('http://') or source.startswith('https://'):
                    # Add authentication if provided
                    if config.get('username') and config.get('password'):
                        parsed = urlparse(source)
                        auth_url = f"{parsed.scheme}://{config['username']}:{config['password']}@{parsed.netloc}{parsed.path}"
                        if parsed.query:
                            auth_url += f"?{parsed.query}"
                        return auth_url
                    return source
                else:
                    # Assume it's an IP address - create HTTP URL
                    # Validate IP format
                    if not self._is_valid_ip_or_hostname(source):
                        raise ValueError(f"Invalid IP address or hostname: {source}")
                    
                    username = config.get('username', '')
                    password = config.get('password', '')
                    auth = f"{username}:{password}@" if username and password else ""
                    return f"http://{auth}{source}/video"
            else:
                raise ValueError("IP camera source must be a non-empty string")
        
        elif camera_type == 'rtsp':
            # RTSP URL format
            if isinstance(source, str) and source.strip():
                source = source.strip()
                if not source.startswith('rtsp://'):
                    # Build RTSP URL
                    if not self._is_valid_ip_or_hostname(source.split('/')[0]):
                        raise ValueError(f"Invalid RTSP source: {source}")
                    
                    username = config.get('username', '')
                    password = config.get('password', '')
                    auth = f"{username}:{password}@" if username and password else ""
                    return f"rtsp://{auth}{source}/stream"
                return source
            else:
                raise ValueError("RTSP source must be a non-empty string")
        
        elif camera_type == 'file':
            if isinstance(source, str) and source.strip():
                return str(source.strip())
            else:
                raise ValueError("File source must be a non-empty string")
        
        return source
    
    def _configure_camera(self, cap: cv2.VideoCapture, config: Dict):
        """Configure camera properties"""
        try:
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get('width', 1280))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get('height', 720))
            
            # Set FPS
            cap.set(cv2.CAP_PROP_FPS, config.get('fps', 30))
            
            # Set buffer size (important for real-time processing)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config.get('buffer_size', 1))
            
            # Additional optimizations for IP cameras
            if config['type'] in ['ip_camera', 'rtsp']:
                # Disable auto exposure for consistent frames
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                
                # Set timeout for network cameras
                if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.get('timeout', 5) * 1000)
                
                # Set read timeout
                if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
                    
        except Exception as e:
            print(f"Warning: Could not configure all camera properties: {e}")
    
    def create_rtsp_url(self,
                       ip_address: str,
                       port: int = 554,
                       username: Optional[str] = None,
                       password: Optional[str] = None,
                       stream_path: str = "/stream") -> str:
        """
        Create RTSP URL for IP cameras
        
        Args:
            ip_address: IP address of the camera
            port: RTSP port (default 554)
            username: Authentication username
            password: Authentication password
            stream_path: Stream path on the camera
            
        Returns:
            Complete RTSP URL
        """
        auth = ""
        if username and password:
            auth = f"{username}:{password}@"
        
        return f"rtsp://{auth}{ip_address}:{port}{stream_path}"
    
    def get_common_camera_urls(self) -> Dict[str, List[str]]:
        """
        Get common camera URL patterns for different manufacturers
        
        Returns:
            Dictionary of manufacturer -> URL patterns
        """
        return {
            'hikvision': [
                'rtsp://{ip}:554/Streaming/Channels/101',
                'rtsp://{ip}:554/h264/ch1/main/av_stream',
                'http://{ip}/ISAPI/Streaming/channels/1/picture'
            ],
            'dahua': [
                'rtsp://{ip}:554/cam/realmonitor?channel=1&subtype=0',
                'http://{ip}/cgi-bin/snapshot.cgi'
            ],
            'axis': [
                'rtsp://{ip}/axis-media/media.amp',
                'http://{ip}/axis-cgi/mjpg/video.cgi'
            ],
            'foscam': [
                'rtsp://{ip}:554/videoMain',
                'http://{ip}/cgi-bin/CGIStream.cgi?cmd=GetMJStream'
            ],
            'generic': [
                'rtsp://{ip}:554/stream',
                'rtsp://{ip}:554/live',
                'http://{ip}/video',
                'http://{ip}/mjpeg',
                'http://{ip}:8080/video'
            ]
        }
    
    def auto_discover_cameras(self, ip_range: str = "192.168.1.") -> List[Dict]:
        """
        Auto-discover IP cameras on the network
        
        Args:
            ip_range: IP range to scan (e.g., "192.168.1.")
            
        Returns:
            List of discovered camera configurations
        """
        discovered = []
        
        # This is a simplified implementation
        # In practice, you'd use more sophisticated network discovery
        common_ports = [80, 554, 8080, 8081]
        
        for i in range(1, 255):
            ip = f"{ip_range}{i}"
            
            # Test common ports (simplified)
            for port in common_ports:
                try:
                    # Quick connection test
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.5)
                    result = sock.connect_ex((ip, port))
                    sock.close()
                    
                    if result == 0:
                        # Port is open, might be a camera
                        camera_info = {
                            'ip': ip,
                            'port': port,
                            'type': 'discovered',
                            'potential_urls': []
                        }
                        
                        # Add potential URLs based on port
                        if port == 554:
                            camera_info['potential_urls'].append(f"rtsp://{ip}:554/stream")
                        elif port in [80, 8080, 8081]:
                            camera_info['potential_urls'].append(f"http://{ip}:{port}/video")
                        
                        discovered.append(camera_info)
                        
                except Exception:
                    continue
        
        return discovered
    
    def test_camera_connection(self, 
                              source: Union[int, str],
                              timeout: int = 5) -> Tuple[bool, Dict]:
        """
        Test camera connection and get properties
        
        Args:
            source: Camera source (index or URL)
            timeout: Connection timeout in seconds
            
        Returns:
            Tuple of (success, properties_dict)
        """
        properties = {}
        
        try:
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                return False, {'error': 'Failed to open camera'}
            
            # Get properties
            properties = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'backend': cap.getBackendName() if hasattr(cap, 'getBackendName') else 'unknown'
            }
            
            # Test frame reading
            start_time = time.time()
            ret, frame = cap.read()
            
            if ret and frame is not None:
                properties['frame_test'] = 'success'
                properties['frame_shape'] = frame.shape
                properties['response_time'] = time.time() - start_time
            else:
                properties['frame_test'] = 'failed'
            
            cap.release()
            return True, properties
            
        except Exception as e:
            return False, {'error': str(e)}
    
    def get_optimal_settings(self, camera_type: str) -> Dict:
        """
        Get optimal settings for different camera types
        
        Args:
            camera_type: Type of camera
            
        Returns:
            Dictionary of optimal settings
        """
        settings = {
            'webcam': {
                'width': 1280,
                'height': 720,
                'fps': 30,
                'buffer_size': 1
            },
            'ip_camera': {
                'width': 1920,
                'height': 1080,
                'fps': 15,
                'buffer_size': 1,
                'timeout': 10
            },
            'rtsp': {
                'width': 1920,
                'height': 1080,
                'fps': 25,
                'buffer_size': 1,
                'timeout': 15,
                'transport': 'tcp'
            }
        }
        
        return settings.get(camera_type, settings['webcam'])
    
    def _is_valid_ip_or_hostname(self, address: str) -> bool:
        """Validate if the given string is a valid IP address or hostname"""
        import socket
        import ipaddress
        
        # Check if it's a valid IP address
        try:
            ipaddress.ip_address(address.split(':')[0])  # Remove port if present
            return True
        except ValueError:
            pass
        
        # Check if it's a valid hostname
        try:
            # Remove port if present
            hostname = address.split(':')[0]
            # Basic hostname validation
            if len(hostname) > 255:
                return False
            if hostname[-1] == ".":
                hostname = hostname[:-1]  # strip exactly one dot from the right, if present
            allowed = re.compile(r"(?!-)[A-Z0-9-]{1,63}(?<!-)$", re.IGNORECASE)
            return all(allowed.match(x) for x in hostname.split("."))
        except Exception:
            return False

# Usage examples and testing
if __name__ == "__main__":
    manager = CameraManager()
    
    # Detect available cameras
    cameras = manager.detect_cameras()
    print("Available cameras:")
    for cam in cameras:
        print(f"  {cam}")
    
    # Test webcam connection
    if cameras:
        success, props = manager.test_camera_connection(0)
        print(f"Webcam test: {success}, {props}")
