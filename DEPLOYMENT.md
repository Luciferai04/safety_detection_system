# Production Deployment Guide

This guide covers deploying the Thermal Power Plant Safety Detection System to production environments.

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ or CentOS 8+
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended
- **Storage**: 50GB minimum, 100GB recommended
- **GPU**: Optional but recommended (NVIDIA with CUDA support)

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.8+
- Git

### Network Requirements
- Ports 80, 443 (HTTP/HTTPS)
- Port 5000 (API)
- Port 7860 (Gradio UI)
- Port 3000 (Grafana)
- Port 9090 (Prometheus)

## Environment Setup

### 1. Install Docker and Docker Compose

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.12.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Clone Repository

```bash
git clone <repository-url>
cd safety_detection_system
```

### 3. Environment Variables

Create a `.env` file with required environment variables:

```bash
# Security
SECRET_KEY=your-secret-key-here
SAFETY_DETECTION_API_KEY=your-api-key-here

# Database
DB_HOST=postgres
DB_NAME=safety_detection
DB_USERNAME=safety_user
DB_PASSWORD=secure-password-here

# Cache
REDIS_HOST=redis
REDIS_PASSWORD=redis-password-here

# External Services
SENTRY_DSN=your-sentry-dsn-here
FACILITY_ID=your-facility-id

# Email Alerts
SMTP_SERVER=smtp.example.com
SMTP_USERNAME=alerts@example.com
SMTP_PASSWORD=smtp-password-here

# Webhooks
WEBHOOK_URL=https://your-webhook-url.com

# Backup
BACKUP_S3_BUCKET=your-backup-bucket

# Monitoring
GRAFANA_PASSWORD=grafana-admin-password
```

### 4. SSL Certificates

For HTTPS support, place your SSL certificates in the `ssl/` directory:

```bash
mkdir ssl
# Copy your certificates:
# ssl/cert.pem
# ssl/key.pem
```

## Deployment Process

### Automated Deployment

Use the provided deployment script:

```bash
# Full production deployment
python deploy.py --environment production

# Skip tests (not recommended)
python deploy.py --environment production --skip-tests

# Skip backup creation
python deploy.py --environment production --skip-backup
```

### Manual Deployment

1. **Build Images**
 ```bash
 docker build -t safety-detection:latest .
 ```

2. **Deploy Services**
 ```bash
 docker-compose -f docker-compose.prod.yml up -d
 ```

3. **Verify Deployment**
 ```bash
 python deploy.py --health-check-only
 ```

## Health Checks

### Automated Health Checks

The deployment script includes comprehensive health checks:

```bash
python deploy.py --health-check-only
```

### Manual Health Checks

1. **API Health**
 ```bash
 curl http://localhost:5000/api/health
 ```

2. **Web Interface**
 ```bash
 curl http://localhost:7860
 ```

3. **Database**
 ```bash
 docker-compose -f docker-compose.prod.yml exec postgres pg_isready
 ```

4. **Cache**
 ```bash
 docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
 ```

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus at `http://localhost:9090`

Key metrics to monitor:
- `safety_detections_total`
- `safety_violations_total`
- `detection_processing_seconds`
- `system_cpu_percent`
- `system_memory_percent`

### Grafana Dashboards

Access Grafana at `http://localhost:3000`
- Username: `admin`
- Password: Set in `.env` file

### Log Aggregation

Logs are available in:
- Application logs: `/var/log/safety_detection/`
- Container logs: `docker-compose logs -f`

### Alerting

Configure alerts for:
- High violation rates
- System resource usage
- Service downtime
- Model prediction errors

## Backup and Recovery

### Automated Backups

Backups are created automatically during deployment:

```bash
# List available backups
ls -la backups/

# Rollback to specific backup
python deploy.py --rollback backups/backup_1640995200
```

### Manual Backup

```bash
# Create backup
docker-compose -f docker-compose.prod.yml exec postgres pg_dump -U safety_user safety_detection > backup.sql

# Backup application data
tar -czf app_backup.tar.gz logs/ models/ data/
```

### Recovery Procedures

1. **Database Recovery**
 ```bash
 # Restore database
 docker-compose -f docker-compose.prod.yml exec -T postgres psql -U safety_user safety_detection < backup.sql
 ```

2. **Application Recovery**
 ```bash
 # Stop services
 docker-compose -f docker-compose.prod.yml down

 # Restore data
 tar -xzf app_backup.tar.gz

 # Restart services
 docker-compose -f docker-compose.prod.yml up -d
 ```

## Security Considerations

### Network Security
- Use HTTPS in production
- Configure firewall rules
- Restrict database access
- Enable container security scanning

### Application Security
- API key authentication enabled
- Rate limiting configured
- Input validation implemented
- Secure secrets management

### Data Protection
- Encrypt data at rest
- Secure data transmission
- Regular security updates
- Access logging enabled

## Performance Optimization

### Resource Limits

Configure resource limits in `docker-compose.prod.yml`:

```yaml
deploy:
 resources:
 limits:
 memory: 4G
 cpus: '2'
 reservations:
 memory: 2G
 cpus: '1'
```

### GPU Acceleration

For GPU-enabled inference:

```yaml
services:
 safety-detection:
 deploy:
 resources:
 reservations:
 devices:
 - driver: nvidia
 count: 1
 capabilities: [gpu]
```

### Caching Strategy

- Redis for session caching
- Model result caching
- Database query optimization
- CDN for static assets

## Troubleshooting

### Common Issues

1. **Container Won't Start**
 ```bash
 # Check logs
 docker-compose -f docker-compose.prod.yml logs safety-detection

 # Check resource usage
 docker stats
 ```

2. **Database Connection Issues**
 ```bash
 # Check database status
 docker-compose -f docker-compose.prod.yml exec postgres pg_isready

 # Verify credentials
 docker-compose -f docker-compose.prod.yml exec postgres psql -U safety_user -d safety_detection
 ```

3. **Model Loading Errors**
 ```bash
 # Check model files
 ls -la models/

 # Verify CUDA availability
 docker-compose -f docker-compose.prod.yml exec safety-detection python -c "import torch; print(torch.cuda.is_available())"
 ```

4. **Performance Issues**
 ```bash
 # Monitor resource usage
 docker stats

 # Check application metrics
 curl http://localhost:5000/api/statistics
 ```

### Log Analysis

```bash
# Application logs
tail -f /var/log/safety_detection/safety_detection.log

# Container logs
docker-compose -f docker-compose.prod.yml logs -f --tail=100 safety-detection

# System logs
journalctl -u docker -f
```

## Updates and Maintenance

### Rolling Updates

1. **Build New Image**
 ```bash
 docker build -t safety-detection:v2.0.0 .
 ```

2. **Update Compose File**
 ```yaml
 image: safety-detection:v2.0.0
 ```

3. **Deploy Update**
 ```bash
 docker-compose -f docker-compose.prod.yml up -d
 ```

### Scheduled Maintenance

- Weekly security updates
- Monthly dependency updates
- Quarterly system optimization
- Annual security audit

### Monitoring Maintenance

- Disk space monitoring
- Log rotation setup
- Backup verification
- Performance baseline updates

## Support and Escalation

### Contact Information
- **Technical Support**: tech-support@company.com
- **Emergency Hotline**: +1-XXX-XXX-XXXX
- **Documentation**: https://docs.company.com/safety-detection

### Escalation Procedures

1. **Level 1**: Application restart
2. **Level 2**: Service rollback
3. **Level 3**: Full system restore
4. **Level 4**: Emergency contact

### Maintenance Windows

- **Preferred**: Weekends 2:00-6:00 AM
- **Emergency**: Any time with approval
- **Notification**: 24 hours advance notice

---

## Deployment Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Backup strategy verified
- [ ] Monitoring alerts configured
- [ ] Security review completed

### During Deployment
- [ ] Services deployed successfully
- [ ] Health checks passing
- [ ] Performance metrics normal
- [ ] Error rates acceptable
- [ ] User acceptance testing completed

### Post-Deployment
- [ ] Monitoring dashboards updated
- [ ] Documentation updated
- [ ] Team notified
- [ ] Backup verified
- [ ] Rollback plan confirmed

---

For additional support or questions, please refer to the main [README.md](README.md) or contact the development team.
