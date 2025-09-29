# Video RSS Aggregator

Video content aggregation and summarization system supporting YouTube, Bilibili (哔哩哔哩), TikTok, Douyin (抖音), and Kuaishou (快手).

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA 11.8+ (optional, for GPU acceleration)
- 16GB+ RAM
- Redis 7.0+
- PostgreSQL 15+
- Python 3.8+
- Node.js 14+

## Architecture

Multi-language implementation optimized for different components:

- **Rust** - RSS server, hardware detection, metadata extraction
- **C++** - Audio processing, transcription engine
- **Python** - ML models for content analysis and summarization
- **Java** - Apache Flink stream processing

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/video-rss-aggregator.git
cd video-rss-aggregator

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Deploy with provided script
./deploy.sh

# Or deploy components manually:
cd distributed-cache
./init-cluster.sh

cd ../model-serving
./deploy.sh

cd ../api-gateway
cargo run --release

cd ../rss-server
cargo run --release
```

## Docker Deployment

```bash
docker-compose -f distributed-cache/docker-compose.yml up -d
docker-compose -f model-serving/docker-compose.triton.yml up -d
docker-compose -f realtime-streaming/docker-compose.kafka.yml up -d
```

## Kubernetes Deployment

```bash
kubectl apply -f model-serving/k8s/
```

## API Access

- GraphQL: http://localhost:8080/graphql
- REST API: http://localhost:8080/api/v3/
- WebSocket: ws://localhost:8080/ws
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## API Usage

### Python SDK
```python
from video_rss_aggregator import Client

client = Client(api_key="your-api-key")
summary = client.summarize_video(url="https://...")
```

### TypeScript SDK
```typescript
import { VideoRSSClient } from '@video-rss-aggregator/sdk';

const client = new VideoRSSClient({ apiKey: 'your-api-key' });
const summary = await client.summarizeVideo({ url: 'https://...' });
```

### REST API
```bash
curl -X POST http://localhost:8080/api/v3/summarize \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=..."}'
```

## Components

- `hardware-detector/` - Hardware detection and configuration
- `video-metadata-extractor/` - Platform API integration
- `audio-processor/` - Audio extraction with optional GPU acceleration
- `transcription-engine/` - Speech-to-text processing
- `video-content-analyzer/` - Video content analysis
- `summarization-engine/` - Text summarization models
- `rss-server/` - RSS feed generation server
- `distributed-cache/` - Redis cluster caching
- `model-serving/` - Model deployment with Triton
- `realtime-streaming/` - WebSocket/WebRTC streaming
- `api-gateway/` - API routing and management
- `security/` - Authentication and authorization
- `content-filter/` - Content moderation
- `performance-monitor/` - Metrics collection

## License

Apache 2.0 - See LICENSE file for details.

## Disclaimer

Please ensure compliance with platform terms of service and copyright laws when using this system.
