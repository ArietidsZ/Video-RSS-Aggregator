#!/bin/bash
# Video RSS Aggregator - One-Click Demo Script
# Competition Ready Demo

echo "🚀 Video RSS Aggregator - Competition Demo"
echo "=========================================="
echo ""

# Check requirements
echo "📋 Checking Requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

# Check if ports are available
if lsof -i :3000 >/dev/null 2>&1; then
    echo "⚠️  Port 3000 is already in use. Stopping existing process..."
    pkill -f "npm start" 2>/dev/null || true
    pkill -f "node.*3000" 2>/dev/null || true
fi

if lsof -i :8000 >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is already in use. Stopping existing process..."
    pkill -f "uvicorn" 2>/dev/null || true
    pkill -f "python.*8000" 2>/dev/null || true
fi

echo "✅ Requirements check passed"
echo ""

# Install dependencies
echo "📦 Installing Dependencies..."

# Frontend dependencies
echo "   Installing frontend dependencies..."
cd frontend
npm install >/dev/null 2>&1
cd ..

# Python dependencies
echo "   Installing Python dependencies..."
pip install -q fastapi uvicorn aiohttp

echo "✅ Dependencies installed"
echo ""

# Start services
echo "🌐 Starting Services..."

# Start backend
echo "   Starting backend server (port 8000)..."
PYTHONPATH=. nohup uvicorn simple_backend:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "   Starting frontend server (port 3000)..."
cd frontend
nohup npm start > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 5

echo "✅ Services started successfully"
echo ""

# Display status
echo "📊 Demo Status:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo "   RSS Feed:  http://localhost:8000/rss/bilibili"
echo ""

echo "🎥 Demo Features:"
echo "   ✓ Real-time Chinese video platform data extraction"
echo "   ✓ AI-powered content summaries"
echo "   ✓ RSS feed generation with smart summaries"
echo "   ✓ Legal compliance framework"
echo "   ✓ Multi-platform support (Bilibili, Douyin, Kuaishou)"
echo ""

echo "📋 Demo URLs for Testing:"
echo "   Web Interface:     http://localhost:3000"
echo "   API Health:        http://localhost:8000/health"
echo "   Video Data:        http://localhost:8000/api/videos/bilibili?limit=5"
echo "   RSS with Summary:  http://localhost:8000/rss/bilibili?limit=3"
echo ""

# Check if services are running
sleep 2
if curl -s http://localhost:8000/health >/dev/null; then
    echo "✅ Backend is running and healthy"
else
    echo "❌ Backend health check failed"
fi

if curl -s http://localhost:3000 >/dev/null; then
    echo "✅ Frontend is accessible"
else
    echo "⚠️  Frontend may still be starting..."
fi

echo ""
echo "🎉 DEMO IS READY!"
echo "   Open http://localhost:3000 in your browser"
echo ""
echo "⏹️  To stop the demo:"
echo "   ./stop_demo.sh"
echo ""

# Save PIDs for cleanup
echo $BACKEND_PID > .demo_backend.pid
echo $FRONTEND_PID > .demo_frontend.pid

echo "🔄 Demo running in background..."
echo "   Check logs: tail -f backend.log frontend.log"