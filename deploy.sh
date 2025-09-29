#!/bin/bash

# Video RSS Aggregator - Production Deployment Script
# Deploys the complete high-performance video processing pipeline

set -e

echo "🚀 Video RSS Aggregator - Production Deployment"
echo "================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install Docker first.${NC}"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose.${NC}"
        exit 1
    fi

    # Check for NVIDIA GPU (optional but recommended)
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo -e "${YELLOW}⚠️  No NVIDIA GPU detected. Performance will be limited.${NC}"
    fi

    echo -e "${GREEN}✅ Prerequisites check complete${NC}"
}

# Deploy distributed cache
deploy_cache() {
    echo -e "${YELLOW}Deploying distributed cache...${NC}"
    cd distributed-cache

    # Initialize Redis Cluster
    if [ -f "init-cluster.sh" ]; then
        ./init-cluster.sh
    else
        docker-compose up -d
    fi

    cd ..
    echo -e "${GREEN}✅ Cache deployed${NC}"
}

# Deploy model serving infrastructure
deploy_models() {
    echo -e "${YELLOW}Deploying model serving infrastructure...${NC}"
    cd model-serving

    if [ -f "deploy.sh" ]; then
        ./deploy.sh
    else
        docker-compose -f docker-compose.triton.yml up -d
    fi

    cd ..
    echo -e "${GREEN}✅ Model serving deployed${NC}"
}

# Deploy streaming infrastructure
deploy_streaming() {
    echo -e "${YELLOW}Deploying real-time streaming...${NC}"
    cd realtime-streaming

    # Start Kafka
    docker-compose -f docker-compose.kafka.yml up -d

    # Deploy streaming services
    if [ -f "deploy-streaming.sh" ]; then
        ./deploy-streaming.sh
    fi

    cd ..
    echo -e "${GREEN}✅ Streaming infrastructure deployed${NC}"
}

# Build and deploy Rust services
deploy_rust_services() {
    echo -e "${YELLOW}Building Rust services...${NC}"

    # Check if Rust is installed
    if ! command -v cargo &> /dev/null; then
        echo -e "${YELLOW}Installing Rust...${NC}"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi

    # Build hardware detector
    echo "Building hardware-detector..."
    cd hardware-detector
    cargo build --release
    cd ..

    # Build RSS server
    echo "Building RSS server..."
    cd rss-server
    cargo build --release
    cd ..

    # Build API gateway
    echo "Building API gateway..."
    cd api-gateway
    cargo build --release
    cd ..

    echo -e "${GREEN}✅ Rust services built${NC}"
}

# Deploy monitoring stack
deploy_monitoring() {
    echo -e "${YELLOW}Deploying monitoring stack...${NC}"
    cd performance-monitor

    # Start Prometheus and Grafana
    docker run -d \
        -p 9090:9090 \
        -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
        --name prometheus \
        prom/prometheus

    docker run -d \
        -p 3000:3000 \
        --name grafana \
        grafana/grafana

    cd ..
    echo -e "${GREEN}✅ Monitoring deployed${NC}"
}

# Start services
start_services() {
    echo -e "${YELLOW}Starting services...${NC}"

    # Start API Gateway
    cd api-gateway
    nohup cargo run --release > ../logs/api-gateway.log 2>&1 &
    cd ..

    # Start RSS Server
    cd rss-server
    nohup cargo run --release > ../logs/rss-server.log 2>&1 &
    cd ..

    echo -e "${GREEN}✅ Services started${NC}"
}

# Health check
health_check() {
    echo -e "${YELLOW}Performing health checks...${NC}"
    sleep 5

    # Check API Gateway
    if curl -s http://localhost:8080/health > /dev/null; then
        echo -e "${GREEN}✅ API Gateway: Healthy${NC}"
    else
        echo -e "${RED}❌ API Gateway: Not responding${NC}"
    fi

    # Check RSS Server
    if curl -s http://localhost:8081/health > /dev/null; then
        echo -e "${GREEN}✅ RSS Server: Healthy${NC}"
    else
        echo -e "${YELLOW}⚠️  RSS Server: May still be starting...${NC}"
    fi

    # Check Grafana
    if curl -s http://localhost:3000 > /dev/null; then
        echo -e "${GREEN}✅ Grafana: Accessible${NC}"
    else
        echo -e "${YELLOW}⚠️  Grafana: May still be starting...${NC}"
    fi
}

# Main deployment flow
main() {
    echo "Starting deployment at $(date)"

    # Create logs directory
    mkdir -p logs

    # Run deployment steps
    check_prerequisites
    deploy_cache
    deploy_models
    deploy_streaming
    deploy_rust_services
    deploy_monitoring
    start_services
    health_check

    echo ""
    echo -e "${GREEN}🎉 Deployment Complete!${NC}"
    echo ""
    echo "Access points:"
    echo "  📊 API Gateway: http://localhost:8080"
    echo "  📡 GraphQL Playground: http://localhost:8080/graphql"
    echo "  📈 Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "  🔍 Prometheus: http://localhost:9090"
    echo ""
    echo "To stop all services, run: ./stop.sh"
}

# Run main deployment
main