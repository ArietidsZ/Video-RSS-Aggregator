# Video RSS Aggregator

Native Rust video RSS aggregator for Chinese platforms (Bilibili, Douyin, Kuaishou) with AI transcription and modern Next.js frontend.

## 🚀 Quick Start

### Backend (Rust)
```bash
cd rust-video-core
cargo run --bin server
```

### Frontend (Next.js)
```bash
cd video-rss-frontend
npm install
npm run dev
```

Access:
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8080
- **RSS Feed**: http://localhost:8080/rss/generate

## 🏗️ Architecture

### Backend (Rust)
- **Server**: Axum web framework with rate limiting and compression
- **Transcription**: Native ONNX Runtime with sherpa-onnx models for Chinese-English speech recognition
- **Database**: SQLx with SQLite, optimized schema with proper indexing
- **Caching**: Redis with connection pooling, ETag support for RSS feeds
- **Resilience**: Circuit breakers, retry logic, and comprehensive error handling
- **Monitoring**: Prometheus metrics, health checks, and performance benchmarks

### Frontend (Next.js)
- **Framework**: Next.js 15 with TypeScript and Tailwind CSS
- **State Management**: React Query for efficient data fetching and caching
- **UI Components**: Responsive design with real-time updates
- **API Integration**: Axios client with automatic error handling and retries

## 📁 Project Structure

```
├── rust-video-core/           # Rust backend
│   ├── src/
│   │   ├── server.rs          # Axum web server
│   │   ├── transcription.rs   # ONNX-based transcription
│   │   ├── cache.rs           # Redis + memory caching
│   │   ├── database.rs        # SQLx database layer
│   │   ├── resilience.rs      # Circuit breakers & retry logic
│   │   ├── benchmarks.rs      # Performance testing
│   │   └── lib.rs             # Core library
│   ├── migrations/            # Database migrations
│   └── static/                # Static HTML dashboard
├── video-rss-frontend/        # Next.js frontend
│   ├── src/
│   │   ├── app/               # Next.js app directory
│   │   ├── components/        # React components
│   │   ├── lib/               # API client & utilities
│   │   └── types/             # TypeScript definitions
│   └── package.json
└── README.md
```

## 🎥 Features

### Video Processing
- Multi-platform data extraction (Bilibili, Douyin, Kuaishou)
- Concurrent video processing with semaphore-based rate limiting
- Smart content deduplication and caching

### AI Transcription
- Native Rust ONNX Runtime integration
- Chinese-English speech recognition with sherpa-onnx models
- Voice Activity Detection (VAD) for better segmentation
- Automatic audio format conversion and resampling

### RSS Generation
- Standards-compliant RSS 2.0 feeds
- ETag-based caching with compression
- Configurable feed options and metadata

### Performance & Reliability
- Redis caching with connection pooling
- Circuit breakers for external API calls
- Comprehensive metrics and monitoring
- Load testing and performance benchmarks

## 🛠️ Development Setup

### Prerequisites
- Rust 1.70+ with Cargo
- Node.js 18+ with npm
- Redis (optional, falls back to memory cache)
- SQLite (included)

### Backend Development
```bash
cd rust-video-core

# Install dependencies
cargo build

# Run with hot reload
cargo watch -x "run --bin server"

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Frontend Development
```bash
cd video-rss-frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check
```

### Database Setup
```bash
cd rust-video-core

# Run migrations
cargo run --bin migrate

# Check database status
sqlite3 database.db ".tables"
```

## 📊 Configuration

### Environment Variables
```bash
# Backend (.env)
DATABASE_URL=sqlite:database.db
REDIS_URL=redis://localhost:6379
LOG_LEVEL=info
SERVER_PORT=8080

# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8080
```

### Platform Settings
Configure rate limits, timeouts, and retry policies through the web interface or directly in the database.

## 🔧 API Usage

### Get Videos
```bash
# Get latest videos
curl "http://localhost:8080/videos?platforms=bilibili&limit=10"

# Search videos
curl "http://localhost:8080/videos?search=technology&sort_by=view_count"
```

### Generate RSS
```bash
# Generate RSS feed
curl -X POST "http://localhost:8080/rss/generate" \
  -H "Content-Type: application/json" \
  -d '{"platforms": ["bilibili", "douyin"]}'

# Get RSS with transcriptions
curl "http://localhost:8080/rss/bilibili?include_transcription=true"
```

### System Stats
```bash
# Get system metrics
curl "http://localhost:8080/stats"

# Health check
curl "http://localhost:8080/health"
```

## 🚀 Deployment

### Production Build
```bash
# Backend
cd rust-video-core
cargo build --release

# Frontend
cd video-rss-frontend
npm run build
```

### Docker (Optional)
```bash
# Build containers
docker build -t video-rss-backend rust-video-core/
docker build -t video-rss-frontend video-rss-frontend/

# Run with docker-compose
docker-compose up
```

## 📈 Performance

- **Transcription**: 4x faster than PyTorch with ONNX optimization
- **Caching**: Sub-millisecond Redis response times
- **Concurrency**: Configurable semaphore limits (default: 1000 concurrent)
- **Rate Limiting**: 100 requests/second with burst capacity
- **Memory**: Efficient memory usage with streaming and connection pooling

## 🧪 Testing

```bash
# Backend tests
cd rust-video-core
cargo test

# Performance benchmarks
cargo bench

# Load testing
cargo test --test integration_tests

# Frontend tests
cd video-rss-frontend
npm test
```

## 🎯 Monitoring

### Metrics Dashboard
- **System Resources**: CPU, memory, disk usage
- **API Performance**: Request rates, response times, error rates
- **Cache Efficiency**: Hit rates, eviction statistics
- **Platform Status**: Availability and response times

### Health Checks
- Database connectivity
- Redis availability
- External platform status
- Model loading status

## 📝 License

Apache License 2.0

## ⚠️ Disclaimer

Educational and research purposes only. Respect platform terms of service and applicable laws.