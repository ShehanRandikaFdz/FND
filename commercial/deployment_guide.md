# Deployment Guide - Commercial Fake News Detector

## 🚀 Production Deployment Overview

This guide covers deploying the commercial Fake News Detector to production environments with enterprise-grade security, scalability, and reliability.

## 🏗️ Architecture Overview

### Production Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Web Servers   │────│   API Gateway   │
│   (nginx/HAProxy)│    │   (Gunicorn)    │    │   (Kong/Envoy)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │────│   Database      │────│   Cache Layer   │
│   (Flask)       │    │   (PostgreSQL)  │    │   (Redis)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Models     │    │   File Storage   │    │   Monitoring    │
│   (TensorFlow)  │    │   (S3/MinIO)    │    │   (Prometheus)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Infrastructure Components
- **Load Balancer**: nginx or HAProxy for traffic distribution
- **Web Servers**: Gunicorn with multiple workers
- **API Gateway**: Kong or Envoy for API management
- **Application**: Flask application with commercial features
- **Database**: PostgreSQL for data persistence
- **Cache**: Redis for session and data caching
- **ML Models**: TensorFlow models for AI processing
- **Storage**: S3 or MinIO for file storage
- **Monitoring**: Prometheus, Grafana, and ELK stack

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - DATABASE_URL=postgresql://user:password@db:5432/factcheck
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=your-secret-key
    depends_on:
      - db
      - redis
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=factcheck
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

## ☁️ Cloud Deployment Options

### AWS Deployment

#### EC2 + RDS + ElastiCache
```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.medium \
    --key-name your-key \
    --security-groups sg-12345678 \
    --user-data file://user-data.sh

# Create RDS database
aws rds create-db-instance \
    --db-instance-identifier factcheck-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --master-username admin \
    --master-user-password your-password \
    --allocated-storage 20

# Create ElastiCache cluster
aws elasticache create-cache-cluster \
    --cache-cluster-id factcheck-redis \
    --cache-node-type cache.t3.micro \
    --engine redis \
    --num-cache-nodes 1
```

#### ECS + Fargate
```yaml
# task-definition.json
{
  "family": "factcheck-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "factcheck-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/factcheck:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:password@db:5432/factcheck"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/factcheck",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### EKS (Kubernetes)
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: factcheck-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: factcheck-app
  template:
    metadata:
      labels:
        app: factcheck-app
    spec:
      containers:
      - name: factcheck-app
        image: your-account.dkr.ecr.region.amazonaws.com/factcheck:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: factcheck-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: factcheck-service
spec:
  selector:
    app: factcheck-app
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

### Google Cloud Platform

#### Cloud Run
```yaml
# cloudbuild.yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/factcheck', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/factcheck']
- name: 'gcr.io/cloud-builders/gcloud'
  args: ['run', 'deploy', 'factcheck', '--image', 'gcr.io/$PROJECT_ID/factcheck', '--region', 'us-central1']
```

#### GKE (Kubernetes)
```yaml
# gke-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: factcheck-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: factcheck-app
  template:
    metadata:
      labels:
        app: factcheck-app
    spec:
      containers:
      - name: factcheck-app
        image: gcr.io/PROJECT_ID/factcheck:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: factcheck-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Azure

#### Container Instances
```bash
# Create resource group
az group create --name factcheck-rg --location eastus

# Create container instance
az container create \
    --resource-group factcheck-rg \
    --name factcheck-app \
    --image your-registry.azurecr.io/factcheck:latest \
    --cpu 1 \
    --memory 2 \
    --ports 5000 \
    --environment-variables \
        DATABASE_URL=postgresql://user:password@db:5432/factcheck
```

#### AKS (Kubernetes)
```yaml
# aks-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: factcheck-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: factcheck-app
  template:
    metadata:
      labels:
        app: factcheck-app
    spec:
      containers:
      - name: factcheck-app
        image: your-registry.azurecr.io/factcheck:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: factcheck-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 🔒 Security Configuration

### SSL/TLS Setup
```nginx
# nginx.conf
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Environment Variables
```bash
# .env.production
FLASK_ENV=production
SECRET_KEY=your-super-secret-key-here
DATABASE_URL=postgresql://user:password@db:5432/factcheck
REDIS_URL=redis://redis:6379/0
STRIPE_SECRET_KEY=sk_live_your_stripe_secret_key
STRIPE_PUBLISHABLE_KEY=pk_live_your_stripe_publishable_key
NEWSAPI_KEY=your_newsapi_key
SENTRY_DSN=your_sentry_dsn
```

### Database Security
```sql
-- Create database user with limited privileges
CREATE USER factcheck_user WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE factcheck TO factcheck_user;
GRANT USAGE ON SCHEMA public TO factcheck_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO factcheck_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO factcheck_user;
```

## 📊 Monitoring and Logging

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'factcheck-app'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "FactCheck Pro Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### ELK Stack Configuration
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    ports:
      - "5044:5044"
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200

volumes:
  elasticsearch_data:
```

## 🚀 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run tests
      run: pytest tests/
    
    - name: Build Docker image
      run: |
        docker build -t factcheck:latest .
        docker tag factcheck:latest your-registry/factcheck:latest
    
    - name: Push to registry
      run: |
        docker push your-registry/factcheck:latest
    
    - name: Deploy to production
      run: |
        # Deploy to your production environment
        kubectl set image deployment/factcheck-app factcheck-app=your-registry/factcheck:latest
```

### GitLab CI
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pip install pytest
    - pytest tests/

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/factcheck-app factcheck-app=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
```

## 📈 Scaling Strategy

### Horizontal Scaling
```yaml
# kubernetes-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: factcheck-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: factcheck-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling
```yaml
# postgresql-replica.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgresql-replica
spec:
  replicas: 2
  selector:
    matchLabels:
      app: postgresql-replica
  template:
    metadata:
      labels:
        app: postgresql-replica
    spec:
      containers:
      - name: postgresql
        image: postgres:13
        env:
        - name: POSTGRES_REPLICATION_MODE
          value: slave
        - name: POSTGRES_MASTER_HOST
          value: postgresql-master
        - name: POSTGRES_REPLICATION_USER
          value: replicator
        - name: POSTGRES_REPLICATION_PASSWORD
          value: replicator_password
```

## 🔧 Maintenance and Updates

### Rolling Updates
```bash
# Update application
kubectl set image deployment/factcheck-app factcheck-app=your-registry/factcheck:v2.0.0

# Check rollout status
kubectl rollout status deployment/factcheck-app

# Rollback if needed
kubectl rollout undo deployment/factcheck-app
```

### Database Migrations
```python
# migrations/001_initial_schema.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('plan', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )

def downgrade():
    op.drop_table('users')
```

### Backup Strategy
```bash
# Database backup
pg_dump -h db-host -U username -d factcheck > backup_$(date +%Y%m%d_%H%M%S).sql

# S3 backup
aws s3 cp backup_$(date +%Y%m%d_%H%M%S).sql s3://your-backup-bucket/

# Automated backup script
#!/bin/bash
BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
pg_dump -h db-host -U username -d factcheck > $BACKUP_FILE
aws s3 cp $BACKUP_FILE s3://your-backup-bucket/
rm $BACKUP_FILE
```

## 🎯 Performance Optimization

### Caching Strategy
```python
# Redis caching
import redis
from flask import Flask

app = Flask(__name__)
redis_client = redis.Redis(host='redis', port=6379, db=0)

@app.route('/analyze')
def analyze():
    # Check cache first
    cache_key = f"analysis:{hash(text)}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return jsonify(json.loads(cached_result))
    
    # Perform analysis
    result = perform_analysis(text)
    
    # Cache result for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return jsonify(result)
```

### Database Optimization
```sql
-- Create indexes for better performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_usage_user_id ON usage_tracking(user_id);
CREATE INDEX idx_usage_date ON usage_tracking(created_at);

-- Partition large tables
CREATE TABLE usage_tracking_2024 PARTITION OF usage_tracking
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### CDN Configuration
```yaml
# CloudFront distribution
Resources:
  CloudFrontDistribution:
    Type: AWS::CloudFront::Distribution
    Properties:
      DistributionConfig:
        Origins:
          - Id: factcheck-origin
            DomainName: your-domain.com
            CustomOriginConfig:
              HTTPPort: 80
              HTTPSPort: 443
              OriginProtocolPolicy: https-only
        DefaultCacheBehavior:
          TargetOriginId: factcheck-origin
          ViewerProtocolPolicy: redirect-to-https
          CachePolicyId: 4135ea2d-6df8-44a3-9df3-4b5a84be39ad
```

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] Code review and testing completed
- [ ] Security scan passed
- [ ] Performance testing completed
- [ ] Database migrations prepared
- [ ] Environment variables configured
- [ ] SSL certificates obtained
- [ ] Monitoring setup configured

### Deployment
- [ ] Backup current production data
- [ ] Deploy new version
- [ ] Run database migrations
- [ ] Verify application health
- [ ] Test critical functionality
- [ ] Monitor for errors
- [ ] Update documentation

### Post-Deployment
- [ ] Monitor application metrics
- [ ] Check error logs
- [ ] Verify user functionality
- [ ] Update team on deployment status
- [ ] Plan next deployment cycle

## 🎯 Conclusion

This deployment guide provides comprehensive instructions for deploying the commercial Fake News Detector to production environments. The guide covers:

- **Multiple deployment options**: Docker, Kubernetes, cloud platforms
- **Security best practices**: SSL/TLS, environment variables, database security
- **Monitoring and logging**: Prometheus, Grafana, ELK stack
- **CI/CD pipelines**: Automated testing and deployment
- **Scaling strategies**: Horizontal and vertical scaling
- **Maintenance procedures**: Updates, backups, migrations

Following this guide will ensure a robust, scalable, and secure production deployment of your commercial Fake News Detector.
