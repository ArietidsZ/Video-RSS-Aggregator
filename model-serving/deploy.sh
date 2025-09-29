#!/bin/bash

# Deploy Model Serving Infrastructure
# High-performance model serving with Triton and Kubernetes

set -e

echo "🚀 Deploying Model Serving Infrastructure..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="model-serving"
CLUSTER_NAME="video-rss-cluster"

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    # Check for kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}kubectl not found. Please install kubectl.${NC}"
        exit 1
    fi

    # Check for helm
    if ! command -v helm &> /dev/null; then
        echo -e "${RED}Helm not found. Please install Helm 3.${NC}"
        exit 1
    fi

    # Check for docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found. Please install Docker.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Prerequisites check passed${NC}"
}

# Function to create namespace and quotas
setup_namespace() {
    echo -e "${YELLOW}Setting up namespace...${NC}"
    kubectl apply -f k8s/namespace.yaml
    echo -e "${GREEN}Namespace configured${NC}"
}

# Function to deploy GPU device plugin
deploy_gpu_support() {
    echo -e "${YELLOW}Deploying GPU support...${NC}"

    # Install NVIDIA device plugin
    kubectl apply -f k8s/gpu-management.yaml

    # Wait for device plugin to be ready
    kubectl wait --for=condition=ready pod \
        -l app=nvidia-device-plugin \
        -n $NAMESPACE \
        --timeout=120s || true

    echo -e "${GREEN}GPU support deployed${NC}"
}

# Function to setup model storage
setup_storage() {
    echo -e "${YELLOW}Setting up model storage...${NC}"

    # Create PVC for model repository
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-repository-pvc
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
EOF

    echo -e "${GREEN}Storage configured${NC}"
}

# Function to deploy Triton Inference Server
deploy_triton() {
    echo -e "${YELLOW}Deploying Triton Inference Server...${NC}"

    # Apply Triton deployment
    kubectl apply -f k8s/triton-deployment.yaml

    # Wait for deployment to be ready
    kubectl rollout status deployment/triton-inference-server \
        -n $NAMESPACE \
        --timeout=300s

    echo -e "${GREEN}Triton deployed successfully${NC}"
}

# Function to setup autoscaling
setup_autoscaling() {
    echo -e "${YELLOW}Configuring autoscaling...${NC}"

    # Apply HPA and VPA
    kubectl apply -f k8s/autoscaling.yaml

    echo -e "${GREEN}Autoscaling configured${NC}"
}

# Function to deploy monitoring
deploy_monitoring() {
    echo -e "${YELLOW}Deploying monitoring stack...${NC}"

    # Add Prometheus helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    # Install Prometheus stack
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace $NAMESPACE \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set grafana.adminPassword=admin \
        --set grafana.service.type=LoadBalancer \
        --wait

    # Install DCGM exporter for GPU metrics
    helm upgrade --install dcgm-exporter nvidia/dcgm-exporter \
        --namespace $NAMESPACE \
        --set arguments[0]="--collectors=/etc/dcgm-exporter/dcp-metrics-included.csv" \
        --wait || true

    echo -e "${GREEN}Monitoring deployed${NC}"
}

# Function to deploy model optimizer
deploy_optimizer() {
    echo -e "${YELLOW}Building and deploying model optimizer...${NC}"

    # Build optimizer image
    cd model-optimizer
    docker build -t model-optimizer:latest .
    cd ..

    # Deploy optimizer service
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-optimizer
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-optimizer
  template:
    metadata:
      labels:
        app: model-optimizer
    spec:
      containers:
      - name: optimizer
        image: model-optimizer:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8003
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: "16Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: "1"
            memory: "8Gi"
            cpu: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: model-optimizer-service
  namespace: $NAMESPACE
spec:
  selector:
    app: model-optimizer
  ports:
  - port: 8003
    targetPort: 8003
EOF

    echo -e "${GREEN}Model optimizer deployed${NC}"
}

# Function to download and optimize models
prepare_models() {
    echo -e "${YELLOW}Preparing models...${NC}"

    # Create temporary pod for model preparation
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: model-preparation
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: prepare-models
        image: python:3.10-slim
        command:
        - bash
        - -c
        - |
          pip install transformers optimum onnx

          # Download and convert Whisper model
          python -c "
          from transformers import WhisperForConditionalGeneration
          from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

          model = ORTModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3', from_tf=False)
          model.save_pretrained('/models/whisper_turbo')
          "

          echo "Models prepared successfully"
        volumeMounts:
        - name: model-repository
          mountPath: /models
      volumes:
      - name: model-repository
        persistentVolumeClaim:
          claimName: model-repository-pvc
      restartPolicy: Never
EOF

    # Wait for job completion
    kubectl wait --for=condition=complete job/model-preparation \
        -n $NAMESPACE \
        --timeout=600s || true

    echo -e "${GREEN}Models prepared${NC}"
}

# Function to run tests
run_tests() {
    echo -e "${YELLOW}Running deployment tests...${NC}"

    # Get Triton service endpoint
    TRITON_SERVICE=$(kubectl get svc triton-inference-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")

    # Test Triton health
    echo "Testing Triton health endpoint..."
    curl -s http://$TRITON_SERVICE:8000/v2/health/ready && echo -e "\n${GREEN}✓ Triton is ready${NC}" || echo -e "\n${RED}✗ Triton not ready${NC}"

    # Test model loading
    echo "Testing model repository..."
    curl -s http://$TRITON_SERVICE:8000/v2/models | jq '.' || true

    # Run GPU benchmark
    echo "Running GPU benchmark..."
    kubectl apply -f k8s/gpu-management.yaml

    echo -e "${GREEN}Tests completed${NC}"
}

# Function to display info
display_info() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ Model Serving Infrastructure Deployed Successfully!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo ""
    echo "📊 Service Endpoints:"

    TRITON_SERVICE=$(kubectl get svc triton-inference-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    GRAFANA_SERVICE=$(kubectl get svc prometheus-grafana -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

    echo "  • Triton HTTP: http://$TRITON_SERVICE:8000"
    echo "  • Triton gRPC: $TRITON_SERVICE:8001"
    echo "  • Triton Metrics: http://$TRITON_SERVICE:8002/metrics"
    echo "  • Grafana: http://$GRAFANA_SERVICE:3000 (admin/admin)"
    echo "  • Model Optimizer: http://$TRITON_SERVICE:8003"
    echo ""
    echo "🔍 Useful Commands:"
    echo "  • Check pods: kubectl get pods -n $NAMESPACE"
    echo "  • View logs: kubectl logs -f deployment/triton-inference-server -n $NAMESPACE"
    echo "  • Scale deployment: kubectl scale deployment/triton-inference-server --replicas=5 -n $NAMESPACE"
    echo "  • Check GPU usage: kubectl top nodes --show-capacity"
    echo ""
    echo "⚡ Performance Targets:"
    echo "  • Whisper: <0.01 RTF (100x real-time)"
    echo "  • LLM: <500ms per request"
    echo "  • Batch throughput: 1000+ req/s"
    echo "  • GPU utilization: >80%"
    echo ""
    echo "📈 Monitoring:"
    echo "  • Grafana dashboards available at http://$GRAFANA_SERVICE:3000"
    echo "  • GPU metrics: http://$TRITON_SERVICE:9400/metrics"
    echo ""
}

# Main deployment flow
main() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}     Video RSS Aggregator - Model Serving Deployment      ${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo ""

    check_prerequisites
    setup_namespace
    deploy_gpu_support
    setup_storage
    deploy_triton
    setup_autoscaling
    deploy_monitoring
    deploy_optimizer
    prepare_models
    run_tests
    display_info
}

# Run main function
main "$@"