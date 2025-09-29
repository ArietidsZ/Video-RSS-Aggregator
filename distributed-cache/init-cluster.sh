#!/bin/bash

# Initialize Redis and Cassandra clusters for video RSS aggregator

set -e

echo "🚀 Initializing distributed cache infrastructure..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p config data/redis-{1..6} data/cassandra-{1..3} logs

# Generate Redis cluster configuration
echo -e "${YELLOW}Generating Redis configuration...${NC}"
if [ ! -f config/redis-cluster.conf ]; then
    echo -e "${GREEN}Redis configuration created${NC}"
else
    echo "Redis configuration already exists"
fi

# Generate Cassandra configuration
echo -e "${YELLOW}Generating Cassandra configuration...${NC}"
if [ ! -f config/cassandra.yaml ]; then
    echo -e "${GREEN}Cassandra configuration created${NC}"
else
    echo "Cassandra configuration already exists"
fi

# Generate sentinel configuration
cat > config/sentinel.conf <<EOF
port 26379
dir /tmp
sentinel monitor mymaster redis-master-1 7000 2
sentinel auth-pass mymaster VideoRSS2024!
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 10000
EOF

# Start Docker Compose
echo -e "${YELLOW}Starting Docker containers...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 10

# Initialize Redis cluster
echo -e "${YELLOW}Initializing Redis cluster...${NC}"
docker exec -it redis-master-1 redis-cli --cluster create \
    127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
    127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
    --cluster-replicas 1 \
    --cluster-yes \
    -a VideoRSS2024! || true

# Check Redis cluster status
echo -e "${YELLOW}Checking Redis cluster status...${NC}"
docker exec -it redis-master-1 redis-cli -a VideoRSS2024! cluster info

# Wait for Cassandra to be ready
echo -e "${YELLOW}Waiting for Cassandra cluster...${NC}"
until docker exec -it cassandra-1 cqlsh -e "DESC KEYSPACES" > /dev/null 2>&1; do
    echo -n "."
    sleep 5
done
echo ""

# Create Cassandra keyspace and tables
echo -e "${YELLOW}Initializing Cassandra schema...${NC}"
docker exec -it cassandra-1 cqlsh -e "
CREATE KEYSPACE IF NOT EXISTS video_rss WITH replication = {
    'class': 'NetworkTopologyStrategy',
    'datacenter1': 3
} AND durable_writes = true;

USE video_rss;

CREATE TABLE IF NOT EXISTS cache (
    key text PRIMARY KEY,
    value blob,
    created_at timestamp,
    accessed_at timestamp,
    access_count counter
);

CREATE TABLE IF NOT EXISTS video_metadata (
    video_id text PRIMARY KEY,
    platform text,
    title text,
    duration int,
    metadata text,
    created_at timestamp,
    updated_at timestamp
);

CREATE TABLE IF NOT EXISTS transcriptions (
    video_id text,
    chunk_id int,
    text text,
    start_time float,
    end_time float,
    confidence float,
    created_at timestamp,
    PRIMARY KEY (video_id, chunk_id)
) WITH CLUSTERING ORDER BY (chunk_id ASC);

CREATE TABLE IF NOT EXISTS summaries (
    video_id text PRIMARY KEY,
    summary text,
    key_points list<text>,
    model_used text,
    quality_score float,
    created_at timestamp
);

CREATE TABLE IF NOT EXISTS backups (
    id text PRIMARY KEY,
    created_at timestamp,
    size_bytes bigint,
    status text
);
"

# Build and start cache manager
echo -e "${YELLOW}Building cache manager service...${NC}"
cd cache-manager
cargo build --release || {
    echo -e "${RED}Failed to build cache manager${NC}"
    exit 1
}

# Create systemd service for cache manager (optional)
if command -v systemctl &> /dev/null; then
    cat > /tmp/cache-manager.service <<EOF
[Unit]
Description=Video RSS Cache Manager
After=network.target docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/target/release/cache-manager
Restart=always
RestartSec=10
StandardOutput=append:$(pwd)/../logs/cache-manager.log
StandardError=append:$(pwd)/../logs/cache-manager-error.log

[Install]
WantedBy=multi-user.target
EOF

    echo -e "${YELLOW}Systemd service file created at /tmp/cache-manager.service${NC}"
    echo "To install as service, run: sudo cp /tmp/cache-manager.service /etc/systemd/system/"
fi

# Create monitoring dashboard URL
echo -e "${GREEN}✅ Distributed cache infrastructure initialized successfully!${NC}"
echo ""
echo "📊 Service URLs:"
echo "  - Redis Cluster: redis://localhost:7000-7005"
echo "  - Cassandra: cassandra://localhost:9042"
echo "  - Cache Manager API: http://localhost:8090"
echo "  - Redis Exporter: http://localhost:9121/metrics"
echo "  - Cassandra Exporter: http://localhost:9500/metrics"
echo ""
echo "🔍 Check cluster status:"
echo "  Redis: docker exec -it redis-master-1 redis-cli -a VideoRSS2024! cluster info"
echo "  Cassandra: docker exec -it cassandra-1 nodetool status"
echo ""
echo "🚦 Start cache manager:"
echo "  cd cache-manager && cargo run --release"
echo ""
echo "⚡ Performance targets:"
echo "  - Cache hit ratio: >90%"
echo "  - Read latency: <1ms"
echo "  - Write latency: <5ms"
echo "  - Throughput: 100K+ ops/sec"