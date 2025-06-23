#!/usr/bin/env python3
"""
Comprehensive Production Test Suite for Safety Detection System

This test suite validates all critical components for production readiness.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
import sys
import time
import threading
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from safety_detector import SafetyDetector
from config_manager import ConfigManager, Environment
from monitoring import MonitoringSystem, MetricsCollector
import api

class TestSafetyDetector:
    """Test suite for SafetyDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create a test detector instance"""
        return SafetyDetector(confidence_threshold=0.5, iou_threshold=0.45)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a 640x480 RGB image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return image
    
    def test_detector_initialization(self):
        """Test detector initialization with various parameters"""
        # Valid initialization
        detector = SafetyDetector(confidence_threshold=0.6, iou_threshold=0.4)
        assert detector.confidence_threshold == 0.6
        assert detector.iou_threshold == 0.4
        assert detector.device in ['cpu', 'cuda', 'mps']
        
        # Invalid confidence threshold
        with pytest.raises(ValueError):
            SafetyDetector(confidence_threshold=1.5)
        
        with pytest.raises(ValueError):
            SafetyDetector(confidence_threshold=0.0)
        
        # Invalid IoU threshold
        with pytest.raises(ValueError):
            SafetyDetector(iou_threshold=1.5)
    
    def test_detection_with_valid_input(self, detector, sample_image):
        """Test detection with valid input"""
        result = detector.detect_safety_equipment(sample_image)
        
        # Check result structure
        assert isinstance(result, dict)
        assert 'detections' in result
        assert 'safety_analysis' in result
        assert 'timestamp' in result
        assert 'processing_time' in result
        
        # Check safety analysis structure
        analysis = result['safety_analysis']
        required_keys = [
            'total_persons', 'persons_with_helmets', 'persons_with_jackets',
            'helmet_compliance_rate', 'jacket_compliance_rate', 
            'overall_compliance_rate', 'violations', 'is_compliant'
        ]
        for key in required_keys:
            assert key in analysis
    
    def test_detection_with_invalid_input(self, detector):
        """Test detection with invalid inputs"""
        # None input
        result = detector.detect_safety_equipment(None)
        assert 'error' in result
        assert 'ValueError' in result['error_type']
        
        # Wrong type
        result = detector.detect_safety_equipment("not an array")
        assert 'error' in result
        assert 'TypeError' in result['error_type']
        
        # Wrong shape
        wrong_shape = np.random.randint(0, 255, (100, 100), dtype=np.uint8)  # 2D instead of 3D
        result = detector.detect_safety_equipment(wrong_shape)
        assert 'error' in result
        
        # Empty array
        empty_array = np.array([])
        result = detector.detect_safety_equipment(empty_array)
        assert 'error' in result
        
        # Extreme dimensions
        too_small = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = detector.detect_safety_equipment(too_small)
        assert 'error' in result
    
    def test_error_tracking(self, detector):
        """Test error tracking functionality"""
        initial_error_count = detector.error_count
        
        # Cause an error
        detector.detect_safety_equipment(None)
        
        assert detector.error_count == initial_error_count + 1
        assert detector.consecutive_errors == 1
        assert detector.last_error is not None
        
        # Cause another error
        detector.detect_safety_equipment("invalid")
        
        assert detector.error_count == initial_error_count + 2
        assert detector.consecutive_errors == 2
        
        # Successful detection should reset consecutive errors
        sample_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_safety_equipment(sample_image)
        
        if 'error' not in result:  # If detection succeeds
            assert detector.consecutive_errors == 0
    
    def test_performance_tracking(self, detector, sample_image):
        """Test performance tracking"""
        initial_times_count = len(detector.processing_times)
        
        # Run detection multiple times
        for _ in range(5):
            result = detector.detect_safety_equipment(sample_image)
            
        # Check that processing times are being tracked
        assert len(detector.processing_times) == initial_times_count + 5
        
        # All processing times should be positive
        assert all(t > 0 for t in detector.processing_times)
    
    def test_draw_detections(self, detector, sample_image):
        """Test detection visualization"""
        # Get detection results
        results = detector.detect_safety_equipment(sample_image)
        
        # Draw detections
        output_image = detector.draw_detections(sample_image, results)
        
        # Check output
        assert output_image.shape == sample_image.shape
        assert output_image.dtype == sample_image.dtype
        
        # Test with error results
        error_results = {'error': 'Test error'}
        output_image = detector.draw_detections(sample_image, error_results)
        assert np.array_equal(output_image, sample_image)


class TestConfigManager:
    """Test suite for ConfigManager class"""
    
    def test_default_initialization(self):
        """Test default configuration initialization"""
        config = ConfigManager(Environment.DEVELOPMENT)
        
        assert config.environment == Environment.DEVELOPMENT
        assert config.model.confidence_threshold == 0.5
        assert config.security.enable_api_key_auth is True
        assert config.logging.level == "INFO"
    
    def test_environment_override(self):
        """Test environment variable overrides"""
        with patch.dict(os.environ, {
            'MODEL_CONFIDENCE_THRESHOLD': '0.7',
            'LOG_LEVEL': 'DEBUG',
            'ENABLE_API_AUTH': 'false'
        }):
            config = ConfigManager(Environment.DEVELOPMENT)
            
            assert config.model.confidence_threshold == 0.7
            assert config.logging.level == "DEBUG"
            assert config.security.enable_api_key_auth is False
    
    def test_validation(self):
        """Test configuration validation"""
        # Valid config should not raise
        config = ConfigManager(Environment.DEVELOPMENT)
        
        # Invalid config should raise
        with patch.dict(os.environ, {'MODEL_CONFIDENCE_THRESHOLD': '1.5'}):
            with pytest.raises(ValueError):
                ConfigManager(Environment.DEVELOPMENT)
    
    def test_api_key_retrieval(self):
        """Test API key retrieval from environment"""
        test_key = "test-api-key-12345"
        
        with patch.dict(os.environ, {'SAFETY_DETECTION_API_KEY': test_key}):
            config = ConfigManager(Environment.DEVELOPMENT)
            assert config.get_api_key() == test_key
    
    def test_production_mode(self):
        """Test production mode configuration"""
        config = ConfigManager(Environment.PRODUCTION)
        
        assert config.is_production() is True
        assert config.is_development() is False


class TestMonitoring:
    """Test suite for monitoring system"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a test metrics collector"""
        return MetricsCollector()
    
    @pytest.fixture
    def monitoring_system(self):
        """Create a test monitoring system"""
        return MonitoringSystem()
    
    def test_metrics_collection(self, metrics_collector):
        """Test metrics collection functionality"""
        # Record some detection metrics
        processing_time = 0.5
        detections = [
            {'equipment_type': 'helmet'},
            {'equipment_type': 'person'}
        ]
        violations = ['Missing helmet detected']
        
        metrics_collector.record_detection(processing_time, detections, violations)
        
        # Check that metrics were recorded
        assert len(metrics_collector.detection_times) == 1
        assert len(metrics_collector.detection_history) == 1
        assert len(metrics_collector.violation_history) == 1
        
        assert metrics_collector.detection_times[0] == processing_time
    
    def test_system_metrics(self, metrics_collector):
        """Test system metrics collection"""
        metrics = metrics_collector.get_system_metrics()
        
        # Check that all required metrics are present
        assert hasattr(metrics, 'cpu_percent')
        assert hasattr(metrics, 'memory_percent')
        assert hasattr(metrics, 'disk_percent')
        
        # Check reasonable values
        assert 0 <= metrics.cpu_percent <= 100
        assert 0 <= metrics.memory_percent <= 100
        assert 0 <= metrics.disk_percent <= 100
    
    def test_detection_metrics(self, metrics_collector):
        """Test detection metrics calculation"""
        # Add some test data
        for i in range(10):
            processing_time = 0.1 + i * 0.05
            detections = [{'equipment_type': 'person'}]
            violations = ['Missing helmet detected'] if i % 3 == 0 else []
            
            metrics_collector.record_detection(processing_time, detections, violations)
        
        detection_metrics = metrics_collector.get_detection_metrics()
        
        assert detection_metrics.total_detections > 0
        assert detection_metrics.average_processing_time > 0
        assert 0 <= detection_metrics.violation_rate <= 100
    
    def test_health_checks(self, monitoring_system):
        """Test health check functionality"""
        # Register a test health check
        def test_check():
            return {'healthy': True, 'message': 'Test check passed'}
        
        monitoring_system.health_checker.register_health_check('test', test_check)
        
        # Run health check
        result = monitoring_system.health_checker.run_health_check('test')
        
        assert result.is_healthy is True
        assert result.component == 'test'
        assert 'Test check passed' in result.message
    
    def test_monitoring_lifecycle(self, monitoring_system):
        """Test monitoring system start/stop"""
        assert monitoring_system.is_running is False
        
        # Start monitoring
        monitoring_system.start_monitoring()
        time.sleep(0.1)  # Allow thread to start
        
        assert monitoring_system.is_running is True
        assert monitoring_system.monitoring_thread is not None
        
        # Stop monitoring
        monitoring_system.stop_monitoring()
        
        assert monitoring_system.is_running is False


class TestAPI:
    """Test suite for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create a test client"""
        api.app.config['TESTING'] = True
        with api.app.test_client() as client:
            yield client
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'timestamp' in data
        assert 'detector_ready' in data
    
    def test_api_documentation(self, client):
        """Test API documentation endpoint"""
        response = client.get('/api/docs')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'title' in data
        assert 'endpoints' in data
    
    @patch('api.detector')
    def test_image_detection_endpoint(self, mock_detector, client):
        """Test image detection endpoint"""
        # Mock detector response
        mock_detector.detect_safety_equipment.return_value = {
            'detections': [],
            'safety_analysis': {
                'total_persons': 0,
                'helmet_compliance_rate': 100,
                'jacket_compliance_rate': 100,
                'violations': [],
                'is_compliant': True
            }
        }
        mock_detector.draw_detections.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(f.name, test_image)
            
            with open(f.name, 'rb') as img_file:
                response = client.post('/api/detect/image', 
                                     data={'image': img_file},
                                     content_type='multipart/form-data')
        
        os.unlink(f.name)
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'success' in data
        assert 'results' in data
    
    def test_error_handling(self, client):
        """Test API error handling"""
        # Test missing image
        response = client.post('/api/detect/image', data={})
        assert response.status_code == 400
        
        # Test invalid endpoint
        response = client.get('/api/nonexistent')
        assert response.status_code == 404


class TestIntegration:
    """Integration tests for complete system"""
    
    def test_end_to_end_detection(self):
        """Test complete detection pipeline"""
        # Initialize components
        detector = SafetyDetector(confidence_threshold=0.5)
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run detection
        results = detector.detect_safety_equipment(test_image)
        
        # Verify results
        if 'error' not in results:
            assert 'detections' in results
            assert 'safety_analysis' in results
            
            # Draw results
            output_image = detector.draw_detections(test_image, results)
            assert output_image.shape == test_image.shape
    
    def test_concurrent_detection(self):
        """Test concurrent detection handling"""
        detector = SafetyDetector(confidence_threshold=0.5)
        results = []
        errors = []
        
        def detection_worker():
            try:
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                result = detector.detect_safety_equipment(test_image)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple concurrent detections
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=detection_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
    
    def test_memory_usage(self):
        """Test memory usage over time"""
        import psutil
        import gc
        
        detector = SafetyDetector(confidence_threshold=0.5)
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run many detections
        for _ in range(100):
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = detector.detect_safety_equipment(test_image)
        
        # Force garbage collection
        gc.collect()
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increase too high: {memory_increase / 1024 / 1024:.1f}MB"


class TestProduction:
    """Production-specific tests"""
    
    def test_configuration_for_production(self):
        """Test production configuration"""
        config = ConfigManager(Environment.PRODUCTION)
        
        # Production should have stricter settings
        assert config.security.enable_api_key_auth is True
        assert config.security.rate_limit_requests <= 100
        assert config.logging.level in ['INFO', 'WARNING', 'ERROR']
        assert config.monitoring.enable_health_checks is True
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        detector = SafetyDetector(confidence_threshold=0.5)
        
        # Cause multiple errors
        for _ in range(5):
            result = detector.detect_safety_equipment(None)
            assert 'error' in result
        
        # System should still work after errors
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_safety_equipment(test_image)
        
        # Should work if model is available
        if detector.model is not None:
            assert 'error' not in result or detector.consecutive_errors < detector.max_consecutive_errors
    
    def test_performance_benchmarks(self):
        """Test performance meets production requirements"""
        detector = SafetyDetector(confidence_threshold=0.5)
        
        # Test processing time
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        processing_times = []
        for _ in range(10):
            start_time = time.time()
            result = detector.detect_safety_equipment(test_image)
            processing_time = time.time() - start_time
            
            if 'error' not in result:
                processing_times.append(processing_time)
        
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            
            # Should process within reasonable time (adjust based on hardware)
            max_processing_time = 5.0  # 5 seconds max
            assert avg_time < max_processing_time, f"Average processing time too high: {avg_time:.2f}s"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
