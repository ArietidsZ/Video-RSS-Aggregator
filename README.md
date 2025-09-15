# Video RSS Aggregator

A modern AI-powered RSS aggregator for Chinese video platforms including Bilibili (哔哩哔哩), Douyin (抖音), and Kuaishou (快手). The system fetches real-time video data, generates AI content summaries that can replace watching videos, and serves RSS feeds.

## 🚀 One-Click Demo Setup

### Universal Setup (Recommended - All Platforms)
```bash
python setup_and_run.py
```

This single command:
- ✅ Works on Windows, macOS, and Linux
- ✅ Checks and installs all dependencies
- ✅ Configures Bilibili settings interactively (optional)
- ✅ Starts all services automatically
- ✅ Opens your browser to the demo

### Alternative Platform-Specific Scripts
- **Windows**: `demo.bat` (double-click or run in CMD)
- **macOS/Linux**: `./demo.sh` (run in Terminal)

## 📋 Prerequisites

### Windows Users
1. **Python 3.8+**: Download from https://www.python.org/downloads/
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" during installation
2. **Node.js 14+**: Download from https://nodejs.org/
3. See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed instructions

### macOS/Linux Users
1. Python 3.8+ (usually pre-installed)
2. Node.js 14+ (`brew install node` on macOS)

## 🎥 Features

- **Real-time Data Extraction**: Fetches latest videos from Bilibili, Douyin, Kuaishou
- **AI Content Summaries**: Generates summaries that replace watching videos
- **RSS Feed Generation**: Standards-compliant RSS 2.0 feeds with rich metadata
- **Legal Compliance**: Fair use academic research framework
- **One-Click Demo**: Competition-ready demo setup

## 🌐 Access Points

Once running:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **RSS Feed**: http://localhost:8000/rss/bilibili

## 📖 Documentation

- [Windows Setup Guide](WINDOWS_SETUP.md)
- [Demo Recording Guide](DEMO_RECORDING_GUIDE.md)
- [API Documentation](http://localhost:8000/docs) (when running)

## 🛠️ Manual Setup

If the one-click demo doesn't work:

### Backend
```bash
# Install Python dependencies
pip install fastapi uvicorn aiohttp

# Start backend
PYTHONPATH=. uvicorn simple_backend:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
# Install dependencies
cd frontend
npm install

# Start frontend
npm start
```

## 🔄 GitHub Sync

For collaborators:
```bash
# Use the sync tool
./sync_github.sh

# Or manually
git pull origin main
git push origin main
```

## 📊 API Examples

### Get Videos
```bash
# Bilibili videos with summaries
curl http://localhost:8000/api/videos/bilibili?limit=5

# RSS feed
curl http://localhost:8000/rss/bilibili?summary=true
```

## 🏆 Competition Information

This project is designed for hackathon/competition demonstration:
1. Run `demo.bat` (Windows) or `./demo.sh` (Unix)
2. Open http://localhost:3000
3. See [DEMO_RECORDING_GUIDE.md](DEMO_RECORDING_GUIDE.md) for presentation tips

## 📝 License

Apache License 2.0 - For educational and personal use only.

## ⚠️ Disclaimer

This tool is for educational purposes and academic research only. Please respect platform terms of service and copyright laws.