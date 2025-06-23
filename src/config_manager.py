"""
Production Configuration Manager
Handles environment-specific configurations, secrets, and validation
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class ModelConfig:
    """Model configuration settings"""
    name: str = "yolov8n"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "auto"
    custom_model_path: Optional[str] = None
    batch_size: int = 16
    max_detections: int = 100
    input_size: int = 640

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    enable_api_key_auth: bool = True
    api_key_header: str = "X-API-Key"
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    allowed_origins: list = None
    max_file_size_mb: int = 16
    allowed_file_types: list = None
    enable_cors: bool = True
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:7860", "http://127.0.0.1:7860"]
        if self.allowed_file_types is None:
            self.allowed_file_types = ["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"]

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    file_path: str = "logs/safety_detection.log"
    max_file_size: str = "10MB"
    backup_count: int = 5
    log_detections: bool = True
    log_violations: bool = True
    log_performance: bool = True
    log_statistics: bool = True
    save_violation_images: bool = True
    violation_images_path: str = "data/violations/"
    enable_structured_logging: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    enable_gpu_acceleration: bool = True
    max_workers: int = 4
    cache_size: int = 100
    frame_skip: int = 1
    enable_model_optimization: bool = True
    memory_limit_gb: Optional[float] = None
    enable_tensorrt: bool = False
    multiprocessing: bool = True
    gpu_memory_limit: str = "4GB"

@dataclass
class MonitoringConfig:
    """System monitoring configuration"""
    enable_health_checks: bool = True
    health_check_interval: int = 30  # seconds
    enable_metrics_collection: bool = True
    metrics_retention_days: int = 30
    alert_violation_threshold: int = 5
    alert_cooldown_seconds: int = 300

class ConfigManager:
    """
    Production-ready configuration manager for the safety detection system
    """
    
    def __init__(self, 
                 environment: Union[str, Environment] = Environment.DEVELOPMENT,
                 config_file: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            environment: Deployment environment
            config_file: Path to configuration file
        """
        self.environment = Environment(environment) if isinstance(environment, str) else environment
        self.config_file = config_file or self._get_default_config_file()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_configuration()
        self._validate_configuration()
        
    def _get_default_config_file(self) -> str:
        """Get default configuration file based on environment"""
        config_dir = Path(__file__).parent.parent / "config"
        
        config_files = {
            Environment.DEVELOPMENT: "config_dev.yaml",
            Environment.STAGING: "config_staging.yaml", 
            Environment.PRODUCTION: "config_prod.yaml",
            Environment.TESTING: "config_test.yaml"
        }
        
        config_file = config_dir / config_files.get(self.environment, "config.yaml")
        
        # Fall back to base config if environment-specific doesn't exist
        if not config_file.exists():
            config_file = config_dir / "config.yaml"
            
        return str(config_file)
    
    def _load_configuration(self):
        """Load configuration from file and environment variables"""
        # Load base configuration
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                base_config = yaml.safe_load(f) or {}
        else:
            self.logger.warning(f"Config file not found: {self.config_file}, using defaults")
            base_config = {}
        
        # Override with environment variables
        env_overrides = self._load_environment_overrides()
        base_config = self._merge_configs(base_config, env_overrides)
        
        # Initialize configuration objects
        self.model = ModelConfig(**base_config.get("model", {}))
        self.security = SecurityConfig(**base_config.get("security", {}))
        self.logging = LoggingConfig(**base_config.get("logging", {}))
        self.performance = PerformanceConfig(**base_config.get("performance", {}))
        self.monitoring = MonitoringConfig(**base_config.get("monitoring", {}))
        
        # Store raw config for access to other settings
        self.raw_config = base_config
        
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables"""
        overrides = {}
        
        # Model configuration
        if os.getenv("MODEL_CONFIDENCE_THRESHOLD"):
            overrides.setdefault("model", {})["confidence_threshold"] = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD"))
        if os.getenv("MODEL_DEVICE"):
            overrides.setdefault("model", {})["device"] = os.getenv("MODEL_DEVICE")
        if os.getenv("MODEL_BATCH_SIZE"):
            overrides.setdefault("model", {})["batch_size"] = int(os.getenv("MODEL_BATCH_SIZE"))
            
        # Security configuration
        if os.getenv("ENABLE_API_AUTH"):
            overrides.setdefault("security", {})["enable_api_key_auth"] = os.getenv("ENABLE_API_AUTH").lower() == "true"
        if os.getenv("API_RATE_LIMIT"):
            overrides.setdefault("security", {})["rate_limit_requests"] = int(os.getenv("API_RATE_LIMIT"))
        if os.getenv("MAX_FILE_SIZE_MB"):
            overrides.setdefault("security", {})["max_file_size_mb"] = int(os.getenv("MAX_FILE_SIZE_MB"))
            
        # Performance configuration
        if os.getenv("MAX_WORKERS"):
            overrides.setdefault("performance", {})["max_workers"] = int(os.getenv("MAX_WORKERS"))
        if os.getenv("ENABLE_GPU"):
            overrides.setdefault("performance", {})["enable_gpu_acceleration"] = os.getenv("ENABLE_GPU").lower() == "true"
            
        # Logging configuration
        if os.getenv("LOG_LEVEL"):
            overrides.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
            
        return overrides
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate model configuration
        if not 0 < self.model.confidence_threshold < 1:
            errors.append("Model confidence threshold must be between 0 and 1")
        if not 0 < self.model.iou_threshold < 1:
            errors.append("Model IoU threshold must be between 0 and 1")
        if self.model.batch_size < 1:
            errors.append("Model batch size must be positive")
            
        # Validate security configuration
        if self.security.rate_limit_requests < 1:
            errors.append("Rate limit requests must be positive")
        if self.security.max_file_size_mb < 1:
            errors.append("Max file size must be positive")
            
        # Validate performance configuration
        if self.performance.max_workers < 1:
            errors.append("Max workers must be positive")
        if self.performance.cache_size < 1:
            errors.append("Cache size must be positive")
            
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_api_key(self) -> Optional[str]:
        """Get API key from environment or configuration"""
        # First check environment variable
        api_key = os.getenv("SAFETY_DETECTION_API_KEY")
        if api_key:
            return api_key
            
        # Then check configuration file (not recommended for production)
        if self.environment != Environment.PRODUCTION:
            return self.raw_config.get("security", {}).get("api_key")
            
        return None
    
    def get_database_url(self) -> Optional[str]:
        """Get database URL from environment"""
        return os.getenv("DATABASE_URL")
    
    def get_redis_url(self) -> Optional[str]:
        """Get Redis URL for caching"""
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_log_level(self) -> int:
        """Get numeric log level"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return level_map.get(self.logging.level.upper(), logging.INFO)
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration as dictionary"""
        return {
            "environment": self.environment.value,
            "model": asdict(self.model),
            "security": asdict(self.security),
            "logging": asdict(self.logging),
            "performance": asdict(self.performance),
            "monitoring": asdict(self.monitoring)
        }
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        if file_path is None:
            file_path = self.config_file
            
        config_data = self.export_config()
        
        with open(file_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
        self.logger.info(f"Configuration saved to {file_path}")
    
    def update_config(self, section: str, **kwargs):
        """Update configuration section with new values"""
        if section == "model":
            for key, value in kwargs.items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        elif section == "security":
            for key, value in kwargs.items():
                if hasattr(self.security, key):
                    setattr(self.security, key, value)
        elif section == "logging":
            for key, value in kwargs.items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
        elif section == "performance":
            for key, value in kwargs.items():
                if hasattr(self.performance, key):
                    setattr(self.performance, key, value)
        elif section == "monitoring":
            for key, value in kwargs.items():
                if hasattr(self.monitoring, key):
                    setattr(self.monitoring, key, value)
        else:
            raise ValueError(f"Unknown configuration section: {section}")
            
        # Re-validate after update
        self._validate_configuration()

# Global configuration instance
_config_manager = None

def get_config_manager(environment: Optional[Union[str, Environment]] = None,
                      config_file: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        # Determine environment
        if environment is None:
            environment = os.getenv("ENVIRONMENT", Environment.DEVELOPMENT.value)
            
        _config_manager = ConfigManager(environment, config_file)
        
    return _config_manager

def reset_config_manager():
    """Reset global configuration manager (for testing)"""
    global _config_manager
    _config_manager = None

# Convenience functions
def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return get_config_manager().model

def get_security_config() -> SecurityConfig:
    """Get security configuration"""
    return get_config_manager().security

def get_logging_config() -> LoggingConfig:
    """Get logging configuration"""
    return get_config_manager().logging

def get_performance_config() -> PerformanceConfig:
    """Get performance configuration"""
    return get_config_manager().performance

def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration"""
    return get_config_manager().monitoring

def is_production() -> bool:
    """Check if running in production"""
    return get_config_manager().is_production()

if __name__ == "__main__":
    # Example usage
    config = ConfigManager(Environment.DEVELOPMENT)
    print("Configuration loaded successfully!")
    print(f"Environment: {config.environment.value}")
    print(f"Model device: {config.model.device}")
    print(f"Security enabled: {config.security.enable_api_key_auth}")
    print(f"Log level: {config.logging.level}")
