#!/usr/bin/env python3
"""
Production Monitoring System for Safety Detection

This module provides comprehensive monitoring, health checks, and metrics collection
for the safety detection system in production environments.
"""

import time
import threading
import psutil
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import os
import sys

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from config_manager import get_config_manager

config = get_config_manager()
logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status data class"""
    is_healthy: bool
    timestamp: datetime
    component: str
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class SystemMetrics:
    """System metrics data class"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_utilization: Optional[float] = None

@dataclass
class DetectionMetrics:
    """Detection performance metrics"""
    total_detections: int
    total_violations: int
    average_processing_time: float
    detections_per_minute: float
    violation_rate: float
    helmet_compliance_rate: float
    jacket_compliance_rate: float

class MetricsCollector:
    """Collects and stores system and application metrics"""
    
    def __init__(self):
        self.detection_times = deque(maxlen=1000)  # Store last 1000 detection times
        self.violation_history = deque(maxlen=10000)  # Store last 10k violations
        self.detection_history = deque(maxlen=10000)  # Store last 10k detections
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self.setup_prometheus_metrics()
        
        # Health check results
        self.health_checks = {}
        self.last_health_check = None
        
        # System performance tracking
        self.performance_alerts = []
        
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.detection_counter = Counter(
            'safety_detections_total',
            'Total number of safety detections',
            ['equipment_type'],
            registry=self.registry
        )
        
        self.violation_counter = Counter(
            'safety_violations_total',
            'Total number of safety violations',
            ['violation_type'],
            registry=self.registry
        )
        
        self.processing_time_histogram = Histogram(
            'detection_processing_seconds',
            'Time spent processing detections',
            registry=self.registry
        )
        
        self.system_cpu_gauge = Gauge(
            'system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_gauge = Gauge(
            'system_memory_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.gpu_memory_gauge = Gauge(
            'gpu_memory_percent',
            'GPU memory usage percentage',
            registry=self.registry
        )
        
        self.health_status_gauge = Gauge(
            'system_health_status',
            'Overall system health status (1=healthy, 0=unhealthy)',
            registry=self.registry
        )
    
    def record_detection(self, processing_time: float, detections: List[Dict], violations: List[str]):
        """Record a detection event"""
        timestamp = datetime.now()
        
        # Store processing time
        self.detection_times.append(processing_time)
        
        # Store detection data
        detection_data = {
            'timestamp': timestamp,
            'processing_time': processing_time,
            'detection_count': len(detections),
            'violation_count': len(violations),
            'violations': violations
        }
        
        self.detection_history.append(detection_data)
        
        # Store violations separately
        for violation in violations:
            self.violation_history.append({
                'timestamp': timestamp,
                'violation_type': violation
            })
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.processing_time_histogram.observe(processing_time)
            
            for detection in detections:
                equipment_type = detection.get('equipment_type', 'unknown')
                self.detection_counter.labels(equipment_type=equipment_type).inc()
            
            for violation in violations:
                self.violation_counter.labels(violation_type=violation).inc()
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_percent=disk.percent
        )
        
        # GPU metrics (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                metrics.gpu_memory_used = gpu_memory.get('allocated_bytes.all.current', 0) / (1024**3)
                metrics.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                metrics.gpu_utilization = torch.cuda.utilization()
        except Exception:
            pass
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.system_cpu_gauge.set(cpu_percent)
            self.system_memory_gauge.set(memory.percent)
            if metrics.gpu_memory_used and metrics.gpu_memory_total:
                gpu_percent = (metrics.gpu_memory_used / metrics.gpu_memory_total) * 100
                self.gpu_memory_gauge.set(gpu_percent)
        
        return metrics
    
    def get_detection_metrics(self) -> DetectionMetrics:
        """Get detection performance metrics"""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        one_hour_ago = now - timedelta(hours=1)
        
        # Recent detections (last hour)
        recent_detections = [d for d in self.detection_history 
                           if d['timestamp'] >= one_hour_ago]
        
        # Recent violations (last hour)
        recent_violations = [v for v in self.violation_history 
                           if v['timestamp'] >= one_hour_ago]
        
        # Detections in last minute
        minute_detections = [d for d in self.detection_history 
                           if d['timestamp'] >= one_minute_ago]
        
        # Calculate metrics
        total_detections = len(recent_detections)
        total_violations = len(recent_violations)
        
        # Average processing time
        avg_processing_time = (sum(self.detection_times) / len(self.detection_times) 
                              if self.detection_times else 0)
        
        # Detections per minute
        detections_per_minute = len(minute_detections)
        
        # Violation rate
        violation_rate = (total_violations / total_detections * 100) if total_detections > 0 else 0
        
        # Compliance rates (simplified calculation)
        helmet_violations = sum(1 for v in recent_violations 
                              if v['violation_type'] == 'Missing helmet detected')
        jacket_violations = sum(1 for v in recent_violations 
                              if v['violation_type'] == 'Missing reflective jacket detected')
        
        helmet_compliance_rate = max(0, 100 - (helmet_violations / max(1, total_detections) * 100))
        jacket_compliance_rate = max(0, 100 - (jacket_violations / max(1, total_detections) * 100))
        
        return DetectionMetrics(
            total_detections=total_detections,
            total_violations=total_violations,
            average_processing_time=avg_processing_time,
            detections_per_minute=detections_per_minute,
            violation_rate=violation_rate,
            helmet_compliance_rate=helmet_compliance_rate,
            jacket_compliance_rate=jacket_compliance_rate
        )
    
    def export_prometheus_metrics(self) -> str:
        """Export Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return "Prometheus not available"
        
        return generate_latest(self.registry)

class HealthChecker:
    """Performs health checks on system components"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.checks = {}
        
    def register_health_check(self, name: str, check_function, interval: int = 30):
        """Register a health check function"""
        self.checks[name] = {
            'function': check_function,
            'interval': interval,
            'last_run': None,
            'last_result': None
        }
    
    def run_health_check(self, name: str) -> HealthStatus:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthStatus(
                is_healthy=False,
                timestamp=datetime.now(),
                component=name,
                message=f"Health check '{name}' not found"
            )
        
        try:
            check = self.checks[name]
            result = check['function']()
            check['last_run'] = datetime.now()
            check['last_result'] = result
            
            return HealthStatus(
                is_healthy=result.get('healthy', False),
                timestamp=datetime.now(),
                component=name,
                message=result.get('message', 'No message'),
                details=result.get('details')
            )
            
        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthStatus(
                is_healthy=False,
                timestamp=datetime.now(),
                component=name,
                message=f"Health check failed: {str(e)}"
            )
    
    def run_all_health_checks(self) -> Dict[str, HealthStatus]:
        """Run all registered health checks"""
        results = {}
        for name in self.checks:
            results[name] = self.run_health_check(name)
        
        # Update overall health status
        overall_health = all(status.is_healthy for status in results.values())
        if PROMETHEUS_AVAILABLE:
            self.metrics_collector.health_status_gauge.set(1 if overall_health else 0)
        
        return results

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
    def check_alert_conditions(self, system_metrics: SystemMetrics, 
                             detection_metrics: DetectionMetrics) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        # High CPU usage
        if system_metrics.cpu_percent > 90:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f'High CPU usage: {system_metrics.cpu_percent:.1f}%',
                'value': system_metrics.cpu_percent
            })
        
        # High memory usage
        if system_metrics.memory_percent > 90:
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f'High memory usage: {system_metrics.memory_percent:.1f}%',
                'value': system_metrics.memory_percent
            })
        
        # High violation rate
        if detection_metrics.violation_rate > config.monitoring.alert_violation_threshold:
            alerts.append({
                'type': 'high_violations',
                'severity': 'critical',
                'message': f'High violation rate: {detection_metrics.violation_rate:.1f}%',
                'value': detection_metrics.violation_rate
            })
        
        # Slow processing
        if detection_metrics.average_processing_time > 5.0:  # 5 seconds
            alerts.append({
                'type': 'slow_processing',
                'severity': 'warning',
                'message': f'Slow detection processing: {detection_metrics.average_processing_time:.1f}s',
                'value': detection_metrics.average_processing_time
            })
        
        return alerts
    
    def send_alert(self, alert: Dict[str, Any]):
        """Send an alert notification"""
        alert_id = f"{alert['type']}_{int(time.time())}"
        
        # Check cooldown
        if alert['type'] in self.active_alerts:
            last_alert_time = self.active_alerts[alert['type']]
            if time.time() - last_alert_time < config.monitoring.alert_cooldown_seconds:
                return  # Still in cooldown
        
        # Log alert
        logger.warning(f"ALERT [{alert['severity'].upper()}]: {alert['message']}")
        
        # Store alert
        alert_data = {
            'id': alert_id,
            'timestamp': datetime.now(),
            'type': alert['type'],
            'severity': alert['severity'],
            'message': alert['message'],
            'value': alert.get('value')
        }
        
        self.alert_history.append(alert_data)
        self.active_alerts[alert['type']] = time.time()
        
        # Send notification (implement based on your needs)
        # For now, just log the alert
        if alert['severity'] == 'critical':
            logger.critical(f"CRITICAL ALERT: {alert['message']}")

class MonitoringSystem:
    """Main monitoring system orchestrator"""
    
    def __init__(self):
        self.config = get_config_manager()
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker(self.metrics_collector)
        self.alert_manager = AlertManager(self.config)
        
        self.monitoring_thread = None
        self.is_running = False
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default health checks"""
        
        def detector_health_check():
            """Check if detector is responsive"""
            try:
                from safety_detector import SafetyDetector
                detector = SafetyDetector()
                # Simple test detection
                import numpy as np
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                result = detector.detect_safety_equipment(test_image)
                
                return {
                    'healthy': 'error' not in result,
                    'message': 'Detector is responsive',
                    'details': {'device': detector.device}
                }
            except Exception as e:
                return {
                    'healthy': False,
                    'message': f'Detector check failed: {str(e)}'
                }
        
        def disk_space_check():
            """Check available disk space"""
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            
            return {
                'healthy': free_gb > 1.0,  # At least 1GB free
                'message': f'Free disk space: {free_gb:.1f}GB',
                'details': {'free_gb': free_gb, 'percent_used': disk.percent}
            }
        
        def memory_check():
            """Check available memory"""
            memory = psutil.virtual_memory()
            
            return {
                'healthy': memory.percent < 95,
                'message': f'Memory usage: {memory.percent:.1f}%',
                'details': {'percent': memory.percent, 'available_gb': memory.available / (1024**3)}
            }
        
        self.health_checker.register_health_check('detector', detector_health_check, 60)
        self.health_checker.register_health_check('disk_space', disk_space_check, 300)
        self.health_checker.register_health_check('memory', memory_check, 30)
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_running:
            logger.warning("Monitoring system is already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Collect metrics
                system_metrics = self.metrics_collector.get_system_metrics()
                detection_metrics = self.metrics_collector.get_detection_metrics()
                
                # Run health checks
                if self.config.monitoring.enable_health_checks:
                    health_results = self.health_checker.run_all_health_checks()
                
                # Check for alerts
                alerts = self.alert_manager.check_alert_conditions(system_metrics, detection_metrics)
                for alert in alerts:
                    self.alert_manager.send_alert(alert)
                
                # Log metrics periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    logger.info(f"System metrics: CPU={system_metrics.cpu_percent:.1f}%, "
                              f"Memory={system_metrics.memory_percent:.1f}%, "
                              f"Violations={detection_metrics.violation_rate:.1f}%")
                
                time.sleep(self.config.monitoring.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        system_metrics = self.metrics_collector.get_system_metrics()
        detection_metrics = self.metrics_collector.get_detection_metrics()
        health_results = self.health_checker.run_all_health_checks()
        
        overall_health = all(status.is_healthy for status in health_results.values())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': overall_health,
            'system_metrics': asdict(system_metrics),
            'detection_metrics': asdict(detection_metrics),
            'health_checks': {name: asdict(status) for name, status in health_results.items()},
            'recent_alerts': list(self.alert_manager.alert_history)[-10:],  # Last 10 alerts
            'monitoring_enabled': self.is_running
        }

# Global monitoring instance
_monitoring_system = None

def get_monitoring_system() -> MonitoringSystem:
    """Get the global monitoring system instance"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringSystem()
    return _monitoring_system

def start_monitoring():
    """Start the global monitoring system"""
    monitoring = get_monitoring_system()
    monitoring.start_monitoring()

def stop_monitoring():
    """Stop the global monitoring system"""
    monitoring = get_monitoring_system()
    monitoring.stop_monitoring()

if __name__ == "__main__":
    # Example usage
    monitoring = MonitoringSystem()
    monitoring.start_monitoring()
    
    try:
        # Simulate some activity
        import time
        import random
        
        for i in range(10):
            # Simulate detection metrics
            processing_time = random.uniform(0.1, 1.0)
            detections = [{'equipment_type': 'helmet'}, {'equipment_type': 'person'}]
            violations = ['Missing helmet detected'] if random.random() < 0.3 else []
            
            monitoring.metrics_collector.record_detection(processing_time, detections, violations)
            
            print(f"Recorded detection {i+1}")
            time.sleep(2)
        
        # Get status report
        report = monitoring.get_status_report()
        print("\nStatus Report:")
        print(json.dumps(report, indent=2, default=str))
        
    except KeyboardInterrupt:
        pass
    finally:
        monitoring.stop_monitoring()
