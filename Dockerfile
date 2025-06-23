# Production Dockerfile for Safety Detection System with CUDA support

# Use NVIDIA CUDA runtime as base image for GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-pip python3.11-dev python3.11-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    libcairo-gobject2 \
    libxcomposite1 \
    libxcursor1 \
    libxi6 \
    libxtst6 \
    libnss3 \
    libcups2 \
    libxss1 \
    libxrandr2 \
    libasound2 \
    libatk1.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r safety && useradd -r -g safety safety

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY run.py .

# Create necessary directories
RUN mkdir -p logs uploads models data /tmp/safety_uploads

# Set ownership to non-root user
RUN chown -R safety:safety /app

# Create log directory with proper permissions
RUN mkdir -p /var/log/safety_detection && \
    chown safety:safety /var/log/safety_detection

# Switch to non-root user
USER safety

# Set environment variables
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=production
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Expose ports
EXPOSE 5000 7860

# Default command (can be overridden)
CMD ["python", "run.py", "--mode", "combined"]
