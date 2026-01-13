# Kwavers PINN API Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Kwavers Physics-Informed Neural Network (PINN) API in production environments. The deployment includes enterprise-grade APIs, monitoring, security, and scalability features.

## Prerequisites

### System Requirements

- **Kubernetes**: 1.24+ cluster with GPU support
- **GPU Nodes**: NVIDIA GPUs with CUDA 12.2+ and cuDNN
- **Storage**: 100GB+ persistent storage for databases and models
- **Network**: Load balancer with SSL termination support

### Software Dependencies

- **PostgreSQL**: 15+ for metadata storage
- **Redis**: 7+ for caching and job queuing
- **Prometheus**: For metrics collection
- **Grafana**: For dashboards and visualization
- **NGINX Ingress Controller**: For API gateway
- **Cert-Manager**: For SSL certificate management

## Quick Start Deployment

### Using Docker Compose (Development)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kwavers/kwavers.git
   cd kwavers
   ```

2. **Start the development stack**:
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment**:
   ```bash
   curl http://localhost:8080/health
   ```

### Using Helm (Production)

1. **Add Helm repository** (if using custom charts):
   ```bash
   helm repo add kwavers https://charts.kwavers.org
   helm repo update
   ```

2. **Install with default configuration**:
   ```bash
   helm install pinn kwavers/kwavers-pinn
   ```

3. **Install with custom values**:
   ```yaml
   # values-custom.yaml
   api:
     replicaCount: 5
   postgresql:
     auth:
       password: "your-secure-password"
   security:
     jwtSecret: "your-jwt-secret"
   ```

   ```bash
   helm install pinn kwavers/kwavers-pinn -f values-custom.yaml
   ```

## Detailed Deployment Steps

### 1. Infrastructure Setup

#### Kubernetes Cluster Preparation

```bash
# Create namespace
kubectl create namespace pinn

# Install NVIDIA GPU operator (if not already installed)
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator \
  --create-namespace

# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace

# Install Cert-Manager
helm repo add cert-manager https://charts.jetstack.io
helm install cert-manager cert-manager/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true
```

#### Database Setup

```bash
# Install PostgreSQL
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgresql bitnami/postgresql \
  --namespace pinn \
  --set auth.postgresPassword="your-postgres-password" \
  --set auth.username="pinn_user" \
  --set auth.password="pinn_password" \
  --set auth.database="pinn_db"
```

#### Redis Setup

```bash
# Install Redis
helm install redis bitnami/redis \
  --namespace pinn \
  --set auth.password="your-redis-password"
```

### 2. Application Deployment

#### Using Helm Chart

```bash
# Deploy PINN API
helm install pinn ./k8s/helm \
  --namespace pinn \
  --set postgresql.auth.password="your-postgres-password" \
  --set redis.auth.password="your-redis-password" \
  --set security.jwtSecret="your-jwt-secret"
```

#### Manual Kubernetes Deployment

```bash
# Create secrets
kubectl create secret generic pinn-secrets \
  --namespace pinn \
  --from-literal=database-url="postgres://pinn_user:pinn_password@postgresql:5432/pinn_db" \
  --from-literal=redis-url="redis://redis:6379" \
  --from-literal=jwt-secret="your-jwt-secret"

# Create configmap
kubectl apply -f k8s/configmap.yaml -n pinn

# Deploy application
kubectl apply -f k8s/deployment.yaml -n pinn
```

### 3. Monitoring Setup

#### Install Prometheus and Grafana

```bash
# Add Prometheus community repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts

# Install Prometheus
helm install prometheus prometheus-community/prometheus \
  --namespace monitoring \
  --create-namespace

# Install Grafana
helm install grafana grafana/grafana \
  --namespace monitoring \
  --set adminPassword="your-admin-password"
```

#### Configure Monitoring

```bash
# Apply Prometheus configuration
kubectl create configmap prometheus-config \
  --namespace monitoring \
  --from-file=prometheus.yml=monitoring/prometheus/prometheus.yml

# Apply alert rules
kubectl create configmap prometheus-alerts \
  --namespace monitoring \
  --from-file=alert_rules.yml=monitoring/prometheus/alert_rules.yml

# Import Grafana dashboard
kubectl create configmap grafana-dashboard \
  --namespace monitoring \
  --from-file=pinn-overview.json=monitoring/grafana/dashboards/pinn-overview.json
```

### 4. Ingress Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pinn-api-ingress
  namespace: pinn
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: pinn-api-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pinn-api-service
            port:
              number: 80
```

```bash
kubectl apply -f ingress.yaml
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `RUST_LOG` | Logging level | `info` | No |
| `DATABASE_URL` | PostgreSQL connection URL | - | Yes |
| `REDIS_URL` | Redis connection URL | - | Yes |
| `JWT_SECRET` | JWT signing secret | - | Yes |
| `API_PORT` | Server port | `8080` | No |
| `API_RATE_LIMIT_ANONYMOUS` | Anonymous requests per minute | `60` | No |
| `API_RATE_LIMIT_AUTHENTICATED` | Authenticated requests per minute | `600` | No |

### Security Configuration

#### JWT Authentication

```bash
# Generate secure JWT secret
openssl rand -hex 32
```

#### API Keys

API keys can be created through the admin API or pre-configured in the database.

#### Rate Limiting

Rate limits are configurable per user type and endpoint. Default limits:
- Anonymous users: 60 requests/minute
- Authenticated users: 600 requests/minute

## API Usage

### Authentication

#### JWT Token

```bash
# Login to get JWT token
curl -X POST http://api.yourdomain.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "password"}'

# Use token in requests
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://api.yourdomain.com/api/v1/models
```

#### API Key

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  http://api.yourdomain.com/api/v1/infer
```

### Training a PINN Model

```bash
curl -X POST http://api.yourdomain.com/api/v1/pinn/train \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "physics_domain": "navier_stokes",
    "geometry": {
      "bounds": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
      "obstacles": [],
      "boundary_conditions": []
    },
    "physics_params": {
      "material_properties": {"viscosity": 0.01},
      "boundary_values": {"inlet_velocity": 1.0},
      "initial_values": {},
      "domain_params": {}
    },
    "training_config": {
      "collocation_points": 10000,
      "batch_size": 64,
      "epochs": 100,
      "learning_rate": 0.001,
      "hidden_layers": [128, 128, 64],
      "adaptive_sampling": true,
      "use_gpu": true
    },
    "callback_url": "https://your-app.com/webhook/training-complete"
  }'
```

### Running Inference

```bash
curl -X POST http://api.yourdomain.com/api/v1/pinn/infer \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_123",
    "coordinates": [[0.1, 0.2, 0.0], [0.2, 0.3, 0.1]],
    "physics_params": {}
  }'
```

## Monitoring and Observability

### Health Checks

```bash
# Overall health
curl http://api.yourdomain.com/health

# Detailed health with dependencies
curl http://api.yourdomain.com/health/detailed
```

### Metrics

```bash
# Prometheus metrics
curl http://api.yourdomain.com/metrics
```

### Dashboards

Access Grafana at `http://grafana.yourdomain.com` with the PINN Overview dashboard for:
- API performance metrics
- Training job monitoring
- GPU utilization tracking
- System resource monitoring
- Error rate analysis

## Troubleshooting

### Common Issues

#### Pod CrashLoopBackOff

```bash
# Check pod logs
kubectl logs -f deployment/pinn-api -n pinn

# Check pod events
kubectl describe pod -l app=kwavers-pinn-api -n pinn
```

#### Database Connection Issues

```bash
# Check database connectivity
kubectl exec -it deployment/postgresql -n pinn -- psql -U pinn_user -d pinn_db

# Verify connection string
kubectl get secret pinn-secrets -n pinn -o yaml
```

#### GPU Issues

```bash
# Check GPU resources
kubectl describe nodes | grep nvidia

# Verify GPU operator installation
kubectl get pods -n gpu-operator
```

### Performance Tuning

#### Scaling the API

```bash
# Scale deployment
kubectl scale deployment pinn-api --replicas=10 -n pinn

# Update HPA configuration
kubectl edit hpa pinn-api-hpa -n pinn
```

#### Database Optimization

```bash
# Check database performance
kubectl exec -it deployment/postgresql -n pinn -- psql -U pinn_user -d pinn_db -c "SELECT * FROM pg_stat_activity;"

# Monitor slow queries
kubectl exec -it deployment/postgresql -n pinn -- psql -U pinn_user -d pinn_db -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"
```

## Backup and Recovery

### Database Backup

```bash
# Create database backup
kubectl exec deployment/postgresql -n pinn -- pg_dump -U pinn_user pinn_db > backup.sql

# Backup to persistent volume
kubectl cp backup.sql pinn/postgresql-0:/tmp/backup.sql
```

### Model Artifacts

```bash
# List model files
kubectl exec deployment/pinn-api -n pinn -- ls -la /app/models/

# Backup models
kubectl cp pinn/pinn-api-0:/app/models ./models-backup
```

## Security Considerations

### Network Security

- All traffic encrypted with TLS 1.3
- Internal service communication via mTLS
- Network policies restricting pod-to-pod communication

### Access Control

- JWT tokens with configurable expiration
- Role-based access control (RBAC)
- API key authentication for service accounts
- Audit logging for all API requests

### Data Protection

- Database encryption at rest
- Sensitive data encrypted in transit
- Regular security updates via automated patching

## Support and Maintenance

### Updating the Deployment

```bash
# Update Helm chart
helm upgrade pinn ./k8s/helm -n pinn

# Rolling update
kubectl rollout restart deployment/pinn-api -n pinn
```

### Log Aggregation

```bash
# View application logs
kubectl logs -f deployment/pinn-api -n pinn

# View system logs
kubectl logs -f deployment/prometheus-server -n monitoring
```

### Contact Information

For support and issues:
- **Documentation**: https://docs.kwavers.org
- **GitHub Issues**: https://github.com/kwavers/kwavers/issues
- **Email Support**: support@kwavers.org

## Appendix

### Helm Chart Values Reference

Complete reference of all configurable Helm chart values in `k8s/helm/values.yaml`.

### API Specification

OpenAPI 3.0 specification available at `/api/v1/spec` endpoint.

### Performance Benchmarks

Expected performance metrics and scaling characteristics documented in the performance optimization summary.
