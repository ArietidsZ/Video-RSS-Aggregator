FROM nvcr.io/nvidia/tensorrt:23.12-py3

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install \
    onnx==1.15.0 \
    onnxruntime-gpu==1.16.3 \
    torch==2.1.2 \
    transformers==4.36.2 \
    optimum==1.16.1 \
    tensorrt==8.6.1 \
    pycuda==2023.1 \
    fastapi==0.109.0 \
    uvicorn==0.26.0 \
    pydantic==2.5.3 \
    prometheus-client==0.19.0 \
    redis==5.0.1 \
    aiofiles==23.2.1

# Create app directory
WORKDIR /app

# Copy application files
COPY optimizer.py /app/
COPY config.yaml /app/
COPY requirements.txt /app/

# Install additional requirements
RUN pip3 install -r requirements.txt

# Create directories
RUN mkdir -p /models /cache /logs

# Set environment variables
ENV CUDA_MODULE_LOADING=LAZY
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the optimizer service
CMD ["uvicorn", "optimizer:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]