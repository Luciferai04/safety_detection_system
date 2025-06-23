#!/usr/bin/env python3
"""
Production Deployment Script for Safety Detection System

This script handles the complete deployment process including:
- Environment validation
- Database migration
- Service configuration
- Health checks
- Rollback capabilities
"""

import os
import sys
import subprocess
import argparse
import logging
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import requests
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeploymentError(Exception):
    """Custom exception for deployment errors"""
    pass

class SafetyDetectionDeployer:
    """Main deployment orchestrator"""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / f"config/config_{environment}.yaml"
        self.docker_compose_file = self.project_root / f"docker-compose.{environment}.yml"
        
        # Load configuration
        self.config = self._load_config()
        
        # Deployment status
        self.deployment_status = {
            'started_at': None,
            'completed_at': None,
            'status': 'pending',
            'services': {},
            'rollback_available': False
        }
    
    def _load_config(self) -> Dict:
        """Load deployment configuration"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {self.config_file}")
                return {}
        except Exception as e:
            raise DeploymentError(f"Failed to load config: {e}")
    
    def _run_command(self, command: List[str], cwd: Optional[Path] = None, 
                    timeout: int = 300) -> subprocess.CompletedProcess:
        """Run a shell command with logging"""
        cwd = cwd or self.project_root
        cmd_str = ' '.join(command)
        
        logger.info(f"Running: {cmd_str}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )
            
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {cmd_str}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"STDERR: {e.stderr}")
            raise DeploymentError(f"Command failed: {cmd_str}")
        
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {cmd_str}")
            raise DeploymentError(f"Command timed out: {cmd_str}")
    
    def validate_environment(self) -> bool:
        """Validate deployment environment and prerequisites"""
        logger.info("Validating deployment environment...")
        
        checks = []
        
        # Check Docker
        try:
            result = self._run_command(['docker', '--version'])
            logger.info(f"Docker version: {result.stdout.strip()}")
            checks.append(("Docker", True))
        except DeploymentError:
            checks.append(("Docker", False))
        
        # Check Docker Compose
        try:
            result = self._run_command(['docker-compose', '--version'])
            logger.info(f"Docker Compose version: {result.stdout.strip()}")
            checks.append(("Docker Compose", True))
        except DeploymentError:
            checks.append(("Docker Compose", False))
        
        # Check required files
        required_files = [
            self.project_root / "Dockerfile",
            self.docker_compose_file,
            self.project_root / "requirements.txt",
            self.project_root / "src" / "safety_detector.py"
        ]
        
        for file_path in required_files:
            exists = file_path.exists()
            checks.append((str(file_path.name), exists))
            if not exists:
                logger.error(f"Required file missing: {file_path}")
        
        # Check environment variables
        required_env_vars = [
            'SECRET_KEY',
            'SAFETY_DETECTION_API_KEY',
            'DB_NAME',
            'DB_USERNAME', 
            'DB_PASSWORD'
        ]
        
        for var in required_env_vars:
            exists = var in os.environ
            checks.append((f"ENV:{var}", exists))
            if not exists:
                logger.warning(f"Environment variable not set: {var}")
        
        # Print validation summary
        logger.info("Environment validation results:")
        for check_name, passed in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {check_name}: {status}")
        
        # Determine if deployment can proceed
        critical_checks = [
            ("Docker", True),
            ("Docker Compose", True),
            ("Dockerfile", True),
            (self.docker_compose_file.name, True)
        ]
        
        can_proceed = all(
            any(check[0] == critical[0] and check[1] == critical[1] 
                for check in checks)
            for critical in critical_checks
        )
        
        if not can_proceed:
            raise DeploymentError("Critical environment validation checks failed")
        
        logger.info("Environment validation completed successfully")
        return True
    
    def backup_current_deployment(self) -> Optional[str]:
        """Backup current deployment for rollback"""
        logger.info("Creating deployment backup...")
        
        try:
            backup_dir = self.project_root / "backups" / f"backup_{int(time.time())}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration
            if self.config_file.exists():
                shutil.copy2(self.config_file, backup_dir / "config.yaml")
            
            # Backup Docker Compose file
            if self.docker_compose_file.exists():
                shutil.copy2(self.docker_compose_file, backup_dir / "docker-compose.yml")
            
            # Export current Docker images
            try:
                result = self._run_command(['docker', 'images', '--format', 'json'])
                images = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
                
                safety_images = [img for img in images if 'safety' in img.get('Repository', '').lower()]
                
                for image in safety_images:
                    repo = image['Repository']
                    tag = image['Tag']
                    image_name = f"{repo}:{tag}"
                    backup_file = backup_dir / f"{repo.replace('/', '_')}_{tag}.tar"
                    
                    logger.info(f"Backing up Docker image: {image_name}")
                    self._run_command(['docker', 'save', '-o', str(backup_file), image_name])
                    
            except Exception as e:
                logger.warning(f"Failed to backup Docker images: {e}")
            
            # Create backup manifest
            manifest = {
                'created_at': time.time(),
                'environment': self.environment,
                'backup_path': str(backup_dir),
                'config_backed_up': (backup_dir / "config.yaml").exists(),
                'compose_backed_up': (backup_dir / "docker-compose.yml").exists()
            }
            
            with open(backup_dir / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Backup created: {backup_dir}")
            self.deployment_status['rollback_available'] = True
            return str(backup_dir)
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None
    
    def build_images(self) -> bool:
        """Build Docker images"""
        logger.info("Building Docker images...")
        
        try:
            # Build main application image
            self._run_command([
                'docker', 'build',
                '-t', 'safety-detection:latest',
                '-f', 'Dockerfile',
                '.'
            ], timeout=600)  # 10 minutes timeout for build
            
            logger.info("Docker image built successfully")
            return True
            
        except DeploymentError as e:
            logger.error(f"Docker build failed: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run test suite before deployment"""
        logger.info("Running test suite...")
        
        try:
            # Install test dependencies
            self._run_command([
                'python', '-m', 'pip', 'install', 
                'pytest', 'pytest-cov'
            ])
            
            # Run tests
            self._run_command([
                'python', '-m', 'pytest', 
                'tests/', 
                '-v',
                '--tb=short',
                '--cov=src',
                '--cov-report=term-missing'
            ])
            
            logger.info("All tests passed")
            return True
            
        except DeploymentError as e:
            logger.error(f"Tests failed: {e}")
            return False
    
    def deploy_services(self) -> bool:
        """Deploy services using Docker Compose"""
        logger.info("Deploying services...")
        
        try:
            # Pull latest images for external services
            self._run_command([
                'docker-compose', 
                '-f', str(self.docker_compose_file),
                'pull', 'postgres', 'redis', 'nginx'
            ])
            
            # Start services
            self._run_command([
                'docker-compose',
                '-f', str(self.docker_compose_file),
                'up', '-d'
            ])
            
            logger.info("Services deployed successfully")
            
            # Wait for services to be ready
            self._wait_for_services()
            
            return True
            
        except DeploymentError as e:
            logger.error(f"Service deployment failed: {e}")
            return False
    
    def _wait_for_services(self, timeout: int = 300) -> bool:
        """Wait for services to be healthy"""
        logger.info("Waiting for services to be ready...")
        
        services_to_check = [
            ('postgres', self._check_postgres),
            ('redis', self._check_redis),
            ('safety-detection', self._check_api)
        ]
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_ready = True
            
            for service_name, check_func in services_to_check:
                if not check_func():
                    all_ready = False
                    logger.info(f"Waiting for {service_name}...")
                    break
                else:
                    self.deployment_status['services'][service_name] = 'ready'
            
            if all_ready:
                logger.info("All services are ready")
                return True
            
            time.sleep(10)
        
        raise DeploymentError("Services failed to become ready within timeout")
    
    def _check_postgres(self) -> bool:
        """Check if PostgreSQL is ready"""
        try:
            result = self._run_command([
                'docker-compose',
                '-f', str(self.docker_compose_file),
                'exec', '-T', 'postgres',
                'pg_isready', '-U', os.getenv('DB_USERNAME', 'postgres')
            ])
            return result.returncode == 0
        except:
            return False
    
    def _check_redis(self) -> bool:
        """Check if Redis is ready"""
        try:
            result = self._run_command([
                'docker-compose',
                '-f', str(self.docker_compose_file),
                'exec', '-T', 'redis',
                'redis-cli', 'ping'
            ])
            return 'PONG' in result.stdout
        except:
            return False
    
    def _check_api(self) -> bool:
        """Check if API is ready"""
        try:
            response = requests.get('http://localhost:5000/api/health', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def run_migrations(self) -> bool:
        """Run database migrations"""
        logger.info("Running database migrations...")
        
        try:
            # Check if migrations are needed
            if not self.config.get('database', {}).get('enabled', False):
                logger.info("Database not enabled, skipping migrations")
                return True
            
            # Run migrations (implement based on your needs)
            # This is a placeholder - implement actual migration logic
            logger.info("No migrations to run")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def health_check(self) -> bool:
        """Perform comprehensive health check"""
        logger.info("Performing health check...")
        
        checks = []
        
        # API health check
        try:
            response = requests.get('http://localhost:5000/api/health', timeout=10)
            api_healthy = response.status_code == 200
            checks.append(('API', api_healthy))
            
            if api_healthy:
                health_data = response.json()
                detector_ready = health_data.get('detector_ready', False)
                checks.append(('Detector', detector_ready))
                
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            checks.append(('API', False))
            checks.append(('Detector', False))
        
        # Web interface health check
        try:
            response = requests.get('http://localhost:7860', timeout=10)
            web_healthy = response.status_code == 200
            checks.append(('Web Interface', web_healthy))
        except Exception as e:
            logger.warning(f"Web interface check failed: {e}")
            checks.append(('Web Interface', False))
        
        # Database health check
        if self.config.get('database', {}).get('enabled', False):
            db_healthy = self._check_postgres()
            checks.append(('Database', db_healthy))
        
        # Cache health check
        cache_healthy = self._check_redis()
        checks.append(('Cache', cache_healthy))
        
        # Print health check results
        logger.info("Health check results:")
        for check_name, healthy in checks:
            status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
            logger.info(f"  {check_name}: {status}")
        
        # All critical services must be healthy
        critical_services = ['API', 'Detector']
        critical_healthy = all(
            healthy for name, healthy in checks 
            if name in critical_services
        )
        
        if not critical_healthy:
            raise DeploymentError("Critical services are not healthy")
        
        logger.info("Health check completed successfully")
        return True
    
    def rollback(self, backup_path: str) -> bool:
        """Rollback to previous deployment"""
        logger.info(f"Rolling back to backup: {backup_path}")
        
        try:
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                raise DeploymentError(f"Backup not found: {backup_path}")
            
            # Load backup manifest
            manifest_file = backup_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                logger.info(f"Rolling back to backup from {manifest['created_at']}")
            
            # Stop current services
            self._run_command([
                'docker-compose',
                '-f', str(self.docker_compose_file),
                'down'
            ])
            
            # Restore configuration files
            if (backup_dir / "config.yaml").exists():
                shutil.copy2(backup_dir / "config.yaml", self.config_file)
                logger.info("Configuration restored")
            
            if (backup_dir / "docker-compose.yml").exists():
                shutil.copy2(backup_dir / "docker-compose.yml", self.docker_compose_file)
                logger.info("Docker Compose file restored")
            
            # Restore Docker images
            for tar_file in backup_dir.glob("*.tar"):
                logger.info(f"Restoring Docker image: {tar_file}")
                self._run_command(['docker', 'load', '-i', str(tar_file)])
            
            # Restart services
            self._run_command([
                'docker-compose',
                '-f', str(self.docker_compose_file),
                'up', '-d'
            ])
            
            # Wait for services
            self._wait_for_services()
            
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def deploy(self, skip_tests: bool = False, skip_backup: bool = False) -> bool:
        """Main deployment process"""
        logger.info(f"Starting deployment to {self.environment}")
        
        self.deployment_status['started_at'] = time.time()
        self.deployment_status['status'] = 'in_progress'
        
        try:
            # Validate environment
            self.validate_environment()
            
            # Create backup
            backup_path = None
            if not skip_backup:
                backup_path = self.backup_current_deployment()
            
            # Run tests
            if not skip_tests:
                if not self.run_tests():
                    raise DeploymentError("Tests failed")
            
            # Build images
            if not self.build_images():
                raise DeploymentError("Image build failed")
            
            # Deploy services
            if not self.deploy_services():
                raise DeploymentError("Service deployment failed")
            
            # Run migrations
            if not self.run_migrations():
                raise DeploymentError("Database migration failed")
            
            # Health check
            if not self.health_check():
                raise DeploymentError("Health check failed")
            
            self.deployment_status['status'] = 'completed'
            self.deployment_status['completed_at'] = time.time()
            
            logger.info("🎉 Deployment completed successfully!")
            
            # Print deployment summary
            duration = self.deployment_status['completed_at'] - self.deployment_status['started_at']
            logger.info(f"Deployment duration: {duration:.1f} seconds")
            
            if backup_path:
                logger.info(f"Rollback available: {backup_path}")
            
            return True
            
        except Exception as e:
            self.deployment_status['status'] = 'failed'
            logger.error(f"❌ Deployment failed: {e}")
            
            # Offer rollback if backup exists
            if backup_path and not skip_backup:
                logger.info("Rollback option available. Use --rollback to restore previous version.")
            
            return False

def main():
    parser = argparse.ArgumentParser(description="Deploy Safety Detection System")
    
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'staging', 'production'],
        default='production',
        help='Deployment environment'
    )
    
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip running tests'
    )
    
    parser.add_argument(
        '--skip-backup',
        action='store_true',
        help='Skip creating backup'
    )
    
    parser.add_argument(
        '--rollback',
        type=str,
        help='Rollback to specified backup path'
    )
    
    parser.add_argument(
        '--health-check-only',
        action='store_true',
        help='Only run health check'
    )
    
    args = parser.parse_args()
    
    deployer = SafetyDetectionDeployer(args.environment)
    
    try:
        if args.rollback:
            success = deployer.rollback(args.rollback)
        elif args.health_check_only:
            success = deployer.health_check()
        else:
            success = deployer.deploy(
                skip_tests=args.skip_tests,
                skip_backup=args.skip_backup
            )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
