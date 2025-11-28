#!/bin/bash
set -e

# Create venv if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
pip install -r ai-service/requirements.txt

# Generate Python Protos
echo "Generating Python code..."
mkdir -p gen/python/video_rss
python -m grpc_tools.protoc -Iproto \
    --python_out=gen/python/video_rss \
    --grpc_python_out=gen/python/video_rss \
    proto/*.proto
touch gen/python/video_rss/__init__.py

echo "Python setup complete."
