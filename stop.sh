#!/bin/bash

# Video RSS Aggregator - Stop All Services

echo "🛑 Stopping Video RSS Aggregator services..."

# Stop Rust services
echo "Stopping Rust services..."
pkill -f "cargo run --release" 2>/dev/null
pkill -f "api-gateway" 2>/dev/null
pkill -f "rss-server" 2>/dev/null

# Stop Docker containers
echo "Stopping Docker containers..."
docker stop prometheus grafana 2>/dev/null
docker rm prometheus grafana 2>/dev/null

# Stop distributed cache
if [ -d "distributed-cache" ]; then
    cd distributed-cache
    docker-compose down
    cd ..
fi

# Stop model serving
if [ -d "model-serving" ]; then
    cd model-serving
    docker-compose -f docker-compose.triton.yml down 2>/dev/null
    cd ..
fi

# Stop streaming services
if [ -d "realtime-streaming" ]; then
    cd realtime-streaming
    docker-compose -f docker-compose.kafka.yml down 2>/dev/null
    cd ..
fi

echo "✅ All services stopped"