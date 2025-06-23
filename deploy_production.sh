#!/bin/bash

# Production deployment script for Safety Detection System with CUDA support
# This script ensures proper setup and deployment of the containerized application

set -e  # Exit on any error

echo "🚀 Starting Safety Detection System Production Deployment with CUDA Support"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if NVIDIA Docker runtime is available
    if ! docker info | grep -q nvidia; then
        print_warning "NVIDIA Docker runtime not detected. GPU acceleration may not work."
        print_warning "Please install nvidia-docker2 and restart Docker daemon."
        print_warning "Run: sudo apt-get install nvidia-docker2 && sudo systemctl restart docker"
    else
        print_success "NVIDIA Docker runtime detected"
    fi
    
    # Check if NVIDIA drivers are installed
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA drivers detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    else
        print_warning "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    fi
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        print_error ".env file not found. Please create it with required environment variables."
        print_status "Creating sample .env file..."
        cat > .env << EOF
# Database Configuration
DB_NAME=safety_detection
DB_USERNAME=safety_user
DB_PASSWORD=your_secure_password_here

# Redis Configuration
REDIS_PASSWORD=your_redis_password_here

# Application Security
SECRET_KEY=your_secret_key_here
SAFETY_DETECTION_API_KEY=your_api_key_here

# Monitoring & Logging
SENTRY_DSN=your_sentry_dsn_here
GRAFANA_PASSWORD=your_grafana_password_here

# Facility Configuration
FACILITY_ID=facility_001

# Email Configuration (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here

# Webhook Configuration (optional)
WEBHOOK_URL=https://your-webhook-url.com/notify

# Backup Configuration (optional)
BACKUP_S3_BUCKET=your-backup-bucket
EOF
        print_warning "Please edit .env file with your actual configuration values before proceeding."
        exit 1
    fi
    
    print_success "Prerequisites check completed"
}

# Test CUDA functionality
test_cuda() {
    print_status "Testing CUDA functionality..."
    
    # Test CUDA with a simple container
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_success "CUDA test passed - GPU access working"
        return 0
    else
        print_warning "CUDA test failed - GPU may not be accessible to containers"
        return 1
    fi
}

# Build and deploy
deploy() {
    print_status "Building and deploying the application..."
    
    # Pull latest images
    print_status "Pulling latest base images..."
    docker-compose -f docker-compose.prod.yml pull postgres redis nginx prometheus grafana
    
    # Build the application
    print_status "Building safety detection application..."
    docker-compose -f docker-compose.prod.yml build --no-cache safety-detection
    
    # Stop existing containers
    print_status "Stopping existing containers..."
    docker-compose -f docker-compose.prod.yml down
    
    # Start the services
    print_status "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_health
}

# Check service health
check_health() {
    print_status "Checking service health..."
    
    # Check if containers are running
    if ! docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        print_error "Some services failed to start"
        docker-compose -f docker-compose.prod.yml logs
        exit 1
    fi
    
    # Check application health endpoint
    max_attempts=10
    attempt=1
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:5000/api/health &> /dev/null; then
            print_success "Application health check passed"
            break
        else
            print_status "Health check attempt $attempt/$max_attempts failed, retrying..."
            sleep 10
            ((attempt++))
        fi
    done
    
    if [ $attempt -gt $max_attempts ]; then
        print_error "Application health check failed after $max_attempts attempts"
        print_status "Checking application logs..."
        docker-compose -f docker-compose.prod.yml logs safety-detection
        exit 1
    fi
    
    # Check GPU usage in container
    print_status "Checking GPU accessibility in container..."
    if docker-compose -f docker-compose.prod.yml exec -T safety-detection nvidia-smi &> /dev/null; then
        print_success "GPU is accessible within the container"
        docker-compose -f docker-compose.prod.yml exec -T safety-detection nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    else
        print_warning "GPU may not be accessible within the container"
    fi
}

# Show deployment status
show_status() {
    print_status "Deployment Status:"
    echo
    docker-compose -f docker-compose.prod.yml ps
    echo
    print_success "🎉 Safety Detection System deployed successfully!"
    echo
    print_status "Service URLs:"
    echo "  • Main Application: http://localhost:5000"
    echo "  • Gradio Interface: http://localhost:7860"
    echo "  • Grafana Dashboard: http://localhost:3000"
    echo "  • Prometheus Metrics: http://localhost:9090"
    echo
    print_status "To view logs: docker-compose -f docker-compose.prod.yml logs -f"
    print_status "To stop services: docker-compose -f docker-compose.prod.yml down"
    echo
}

# Cleanup function
cleanup() {
    if [ $? -ne 0 ]; then
        print_error "Deployment failed!"
        print_status "Cleaning up..."
        docker-compose -f docker-compose.prod.yml down
        print_status "Check logs with: docker-compose -f docker-compose.prod.yml logs"
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Main deployment flow
main() {
    echo "========================================"
    echo "Safety Detection System - Production Deploy"
    echo "========================================"
    echo
    
    check_prerequisites
    echo
    
    if test_cuda; then
        print_success "CUDA support verified - proceeding with GPU-accelerated deployment"
    else
        print_warning "CUDA support not verified - deployment will continue but may run on CPU"
        read -p "Continue with deployment? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Deployment cancelled"
            exit 0
        fi
    fi
    echo
    
    deploy
    echo
    
    show_status
}

# Run main function
main "$@"
