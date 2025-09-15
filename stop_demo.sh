#!/bin/bash
# Stop Demo Script

echo "⏹️  Stopping Video RSS Aggregator Demo..."

# Kill processes using PIDs if available
if [ -f .demo_backend.pid ]; then
    BACKEND_PID=$(cat .demo_backend.pid)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID
        echo "   ✓ Stopped backend (PID: $BACKEND_PID)"
    fi
    rm -f .demo_backend.pid
fi

if [ -f .demo_frontend.pid ]; then
    FRONTEND_PID=$(cat .demo_frontend.pid)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID
        echo "   ✓ Stopped frontend (PID: $FRONTEND_PID)"
    fi
    rm -f .demo_frontend.pid
fi

# Fallback: kill by process name
pkill -f "uvicorn simple_backend" 2>/dev/null && echo "   ✓ Stopped uvicorn processes"
pkill -f "npm start" 2>/dev/null && echo "   ✓ Stopped npm processes"

# Clean up log files
rm -f backend.log frontend.log

echo "✅ Demo stopped successfully"
echo "   All processes terminated and cleaned up"