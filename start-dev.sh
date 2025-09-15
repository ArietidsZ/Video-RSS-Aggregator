#!/bin/bash

# Video RSS Aggregator Development Server Launcher
# This script starts both the backend API and frontend development servers

set -e

echo "🚀 Starting Video RSS Aggregator Development Environment"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed"
    exit 1
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to install Python dependencies
install_python_deps() {
    echo "📦 Installing Python dependencies..."
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi

    source venv/bin/activate
    pip install -r requirements.txt
    echo "✅ Python dependencies installed"
}

# Function to install Node.js dependencies
install_node_deps() {
    echo "📦 Installing Node.js dependencies..."
    cd frontend
    npm install
    cd ..
    echo "✅ Node.js dependencies installed"
}

# Function to start backend
start_backend() {
    echo "🔧 Starting Backend API server..."
    source venv/bin/activate

    # Set environment variables
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    export DEBUG=True
    export LOG_LEVEL=INFO

    # Start the FastAPI server
    python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo "🌐 Backend API started on http://localhost:8000 (PID: $BACKEND_PID)"
    echo "📚 API Documentation: http://localhost:8000/docs"
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting Frontend development server..."
    cd frontend

    # Set environment variables
    export REACT_APP_API_URL=http://localhost:8000
    export PORT=3000

    # Start the React development server
    npm start &
    FRONTEND_PID=$!
    cd ..
    echo "🖥️  Frontend started on http://localhost:3000 (PID: $FRONTEND_PID)"
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down development servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        echo "✅ Backend server stopped"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "✅ Frontend server stopped"
    fi
    echo "👋 Development environment stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    # Check if ports are available
    if ! check_port 8000; then
        echo "❌ Backend port 8000 is in use. Please stop the service using this port."
        exit 1
    fi

    if ! check_port 3000; then
        echo "❌ Frontend port 3000 is in use. Please stop the service using this port."
        exit 1
    fi

    # Install dependencies
    echo "🔍 Checking dependencies..."
    install_python_deps
    install_node_deps

    echo ""
    echo "🏁 Starting development servers..."
    echo "=================================="

    # Start backend
    start_backend
    sleep 3  # Give backend time to start

    # Start frontend
    start_frontend
    sleep 3  # Give frontend time to start

    echo ""
    echo "🎉 Development environment is ready!"
    echo "======================================"
    echo "🌐 Backend API: http://localhost:8000"
    echo "📚 API Docs:    http://localhost:8000/docs"
    echo "🖥️  Frontend:    http://localhost:3000"
    echo "📊 Health:      http://localhost:8000/health"
    echo ""
    echo "📱 Quick Links:"
    echo "   • Dashboard:  http://localhost:3000/"
    echo "   • Search:     http://localhost:3000/search"
    echo "   • RSS Feeds:  http://localhost:3000/feeds"
    echo "   • Settings:   http://localhost:3000/settings"
    echo ""
    echo "🛠️  Testing RSS Feeds:"
    echo "   • Bilibili:   http://localhost:8000/rss/bilibili"
    echo "   • Douyin:     http://localhost:8000/rss/douyin"
    echo "   • Kuaishou:   http://localhost:8000/rss/kuaishou"
    echo "   • Combined:   http://localhost:8000/rss/all"
    echo ""
    echo "⚡ The servers will auto-reload when you make changes"
    echo "🛑 Press Ctrl+C to stop both servers"
    echo ""

    # Wait for user to stop
    while true; do
        sleep 1
    done
}

# Run main function
main