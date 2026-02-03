# MalVec Deployment Guide

This guide covers production deployment of MalVec for malware classification.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Security Considerations](#security-considerations)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Backup and Recovery](#backup-and-recovery)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 20 GB SSD | 50+ GB SSD |
| GPU | None | CUDA-capable (optional) |

### Software Requirements

- Python 3.11+
- Linux (Ubuntu 22.04+), macOS 13+, or Windows Server 2022
- Docker 24+ (for containerized deployment)
- Kubernetes 1.28+ (for orchestrated deployment)

---

## Installation

### Standard Installation

```bash
# Clone repository
git clone https://github.com/yourusername/malvec
cd malvec

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for better UX
pip install rich pyyaml

# Install MalVec
pip install -e .

# Verify installation
python -m malvec --version
```

### System Service (Linux)

Create `/etc/systemd/system/malvec.service`:

```ini
[Unit]
Description=MalVec Malware Classification Service
After=network.target

[Service]
Type=simple
User=malvec
Group=malvec
WorkingDirectory=/opt/malvec
ExecStart=/opt/malvec/venv/bin/python -m malvec.api.server
Restart=always
RestartSec=5
Environment="MALVEC_MODEL_PATH=/opt/malvec/model"
Environment="MALVEC_AUDIT_LOG=/var/log/malvec/audit.log"

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/log/malvec /tmp/malvec

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable malvec
sudo systemctl start malvec
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir rich pyyaml

# Copy application
COPY . .
RUN pip install -e .

# Create non-root user
RUN useradd -m -r malvec
USER malvec

# Create directories
RUN mkdir -p /home/malvec/model /home/malvec/logs

# Environment
ENV MALVEC_MODEL_PATH=/home/malvec/model
ENV MALVEC_AUDIT_LOG=/home/malvec/logs/audit.log

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import malvec; print('healthy')" || exit 1

# Default command
CMD ["python", "-m", "malvec.api.server"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  malvec:
    build: .
    container_name: malvec
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./model:/home/malvec/model:ro
      - malvec-logs:/home/malvec/logs
    environment:
      - MALVEC_VERBOSE=false
      - MALVEC_TIMEOUT=60
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:size=1G,mode=1777

volumes:
  malvec-logs:
```

### Build and Run

```bash
# Build image
docker build -t malvec:latest .

# Run container
docker-compose up -d

# View logs
docker-compose logs -f malvec

# Stop
docker-compose down
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: malvec
  labels:
    app: malvec
spec:
  replicas: 3
  selector:
    matchLabels:
      app: malvec
  template:
    metadata:
      labels:
        app: malvec
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: malvec
        image: malvec:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: MALVEC_MODEL_PATH
          value: "/model"
        - name: MALVEC_AUDIT_LOG
          value: "/logs/audit.log"
        volumeMounts:
        - name: model
          mountPath: /model
          readOnly: true
        - name: logs
          mountPath: /logs
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
              - ALL
      volumes:
      - name: model
        persistentVolumeClaim:
          claimName: malvec-model
      - name: logs
        emptyDir: {}
```

### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: malvec
spec:
  selector:
    app: malvec
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
```

### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: malvec-config
data:
  config.yaml: |
    model_path: /model
    embedding_dim: 384
    k: 5
    confidence_threshold: 0.7
    sandbox_enabled: true
    timeout: 30
    max_memory: 536870912
    max_filesize: 52428800
    verbose: false
```

### Deploy

```bash
# Create namespace
kubectl create namespace malvec

# Apply manifests
kubectl apply -f k8s/ -n malvec

# Check status
kubectl get pods -n malvec

# View logs
kubectl logs -f deployment/malvec -n malvec
```

---

## Security Considerations

### Network Isolation

MalVec should run in an isolated network segment:

```bash
# Docker network isolation
docker network create --internal malvec-internal
```

### File System Protection

- Mount model directory as read-only
- Use tmpfs for temporary files
- Enable read-only root filesystem

### Resource Limits

Always set resource limits to prevent DoS:

```yaml
resources:
  limits:
    memory: "8Gi"
    cpu: "4"
```

### Audit Logging

Ensure audit logs are:
- Written to persistent storage
- Rotated to prevent disk exhaustion
- Shipped to SIEM for analysis

```bash
# Log rotation (Linux)
cat > /etc/logrotate.d/malvec << EOF
/var/log/malvec/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 malvec malvec
}
EOF
```

### VM Isolation (Recommended)

For highest security, run MalVec in an isolated VM:

1. Create dedicated VM with minimal OS
2. Disable network (or restrict to API only)
3. Snapshot before each analysis batch
4. Restore to clean state periodically

---

## Monitoring and Alerting

### Metrics

Key metrics to monitor:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `malvec_classifications_total` | Total classifications | - |
| `malvec_classification_latency_ms` | Processing time | > 5000ms |
| `malvec_sandbox_violations_total` | Security violations | > 0 |
| `malvec_errors_total` | Processing errors | > 10/min |
| `malvec_queue_depth` | Pending items | > 100 |

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'malvec'
    static_configs:
      - targets: ['malvec:8080']
    scrape_interval: 15s
```

### Alertmanager Rules

```yaml
# alerts.yml
groups:
- name: malvec
  rules:
  - alert: MalVecHighLatency
    expr: malvec_classification_latency_ms > 5000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "MalVec classification latency high"

  - alert: MalVecSandboxViolation
    expr: increase(malvec_sandbox_violations_total[5m]) > 0
    labels:
      severity: critical
    annotations:
      summary: "MalVec sandbox violation detected"
```

### Grafana Dashboard

Import the MalVec dashboard JSON from `monitoring/grafana-dashboard.json`.

---

## Backup and Recovery

### Model Backup

```bash
# Create archive backup
python -m malvec.cli.archive create \
    --model ./model \
    --output backups/model_$(date +%Y%m%d).malvec

# Verify backup
python -m malvec.cli.archive inspect backups/model_$(date +%Y%m%d).malvec
```

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/malvec"
MODEL_DIR="/opt/malvec/model"
RETENTION_DAYS=30

# Create backup
BACKUP_FILE="${BACKUP_DIR}/model_$(date +%Y%m%d_%H%M%S).malvec"
python -m malvec.cli.archive create --model "$MODEL_DIR" --output "$BACKUP_FILE"

# Cleanup old backups
find "$BACKUP_DIR" -name "*.malvec" -mtime +$RETENTION_DAYS -delete

echo "Backup complete: $BACKUP_FILE"
```

### Recovery

```bash
# Extract model from backup
python -m malvec.cli.archive extract \
    backups/model_20240115.malvec \
    --output ./model

# Verify model
python -m malvec.cli.classify --model ./model --file test.exe
```

---

## Performance Tuning

### CPU Optimization

```bash
# Use all available cores for FAISS
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
```

### Memory Optimization

For large models, increase memory limits:

```yaml
# config.yaml
max_memory: 1073741824  # 1GB for extraction
```

### GPU Acceleration

If CUDA is available:

```bash
# Install GPU-enabled FAISS
pip install faiss-gpu

# Verify GPU
python -c "import faiss; print(faiss.get_num_gpus())"
```

### Batch Processing

For high-throughput scenarios:

```bash
# Parallel batch processing
python -m malvec.cli.batch \
    --model ./model \
    --input-dir ./samples \
    --output results.csv \
    --parallel 8
```

---

## Troubleshooting

### Common Issues

#### Container won't start

```bash
# Check logs
docker logs malvec

# Common causes:
# - Model not mounted correctly
# - Insufficient memory
# - Permission issues
```

#### High memory usage

```bash
# Monitor memory
docker stats malvec

# Solutions:
# - Reduce batch size
# - Increase container memory limit
# - Use streaming processing
```

#### Slow classification

```bash
# Check if GPU is being used
python -c "import faiss; print('GPU:', faiss.get_num_gpus())"

# Solutions:
# - Enable GPU acceleration
# - Reduce embedding dimension
# - Use smaller K value
```

### Health Check

```bash
# CLI health check
python -c "
from malvec.classifier import KNNClassifier
clf = KNNClassifier.load('./model/model')
print('Model loaded successfully')
print(f'Samples: {clf.n_samples}')
"
```

### Support

For issues:
1. Check audit logs: `/var/log/malvec/audit.log`
2. Review documentation: [docs/](.)
3. Open GitHub issue with logs and config

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0 | Initial deployment guide |

---

## References

- [User Guide](USER_GUIDE.md)
- [Security Architecture](SECURITY.md)
- [API Reference](API.md)
