#!/usr/bin/env python3
"""
Ultra-Optimized Video RSS Aggregator Backend
Maximum performance with advanced parallelism and batching
"""

import asyncio
import os
import sys
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import deque
import numpy as np

# Set UTF-8 encoding
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import uvicorn

# Performance monitoring
METRICS = {
    'total_requests': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'avg_latency': 0,
    'batch_queue': deque(maxlen=100),
    'processing_times': deque(maxlen=100)
}

# Global model instances
GLOBAL_MODELS = {
    'whisper': None,
    'qwen': None,
    'tokenizer': None,
    'device': None,
    'initialized': False,
    'lock': threading.Lock(),
    'batch_lock': threading.Lock()
}

# Advanced caching with LRU
TRANSCRIPTION_CACHE = {}
CACHE_DURATION = timedelta(hours=12)
MAX_CACHE_SIZE = 1000

# Thread pools for different tasks
CPU_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="cpu")
IO_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="io")
GPU_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu")

# Batch processing queue
BATCH_QUEUE = asyncio.Queue(maxsize=100)
BATCH_SIZE = 4
BATCH_TIMEOUT = 0.1  # 100ms

class UltraOptimizedTranscriber:
    """Ultra-optimized transcriber with batching and advanced parallelism"""

    @classmethod
    async def initialize_global_models(cls):
        """Initialize models with optimization flags"""
        with GLOBAL_MODELS['lock']:
            if GLOBAL_MODELS['initialized']:
                return

            try:
                import torch
                import whisper
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True

                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[STARTUP] Initializing models on {device} with optimizations...")

                # Load Whisper with fp16
                print("[STARTUP] Loading Whisper model (fp16)...")
                GLOBAL_MODELS['whisper'] = whisper.load_model("base", device=device)
                if device == "cuda":
                    GLOBAL_MODELS['whisper'] = GLOBAL_MODELS['whisper'].half()

                # Load Qwen with aggressive quantization
                print("[STARTUP] Loading Qwen2.5-7B-Instruct (4-bit optimized)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                model_name = "Qwen/Qwen2.5-7B-Instruct"
                GLOBAL_MODELS['tokenizer'] = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=True  # Use fast tokenizer
                )

                GLOBAL_MODELS['qwen'] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )

                # Enable model optimizations
                if hasattr(GLOBAL_MODELS['qwen'], 'eval'):
                    GLOBAL_MODELS['qwen'].eval()

                GLOBAL_MODELS['device'] = device
                GLOBAL_MODELS['initialized'] = True

                # Extensive warmup
                print("[STARTUP] Warming up models with multiple samples...")
                await cls.extensive_warmup()

                print("[STARTUP] ✅ Ultra-optimized models ready!")

            except Exception as e:
                print(f"[ERROR] Model initialization failed: {e}")
                raise

    @classmethod
    async def extensive_warmup(cls):
        """Extensive warmup for optimal performance"""
        try:
            import torch
            import numpy as np

            # Warmup Whisper with various sizes
            if GLOBAL_MODELS['whisper']:
                for size in [8000, 16000, 32000]:
                    sample = np.zeros(size, dtype=np.float32)
                    with torch.no_grad():
                        GLOBAL_MODELS['whisper'].transcribe(sample, language='zh', fp16=True)

            # Warmup Qwen with various prompts
            if GLOBAL_MODELS['qwen'] and GLOBAL_MODELS['tokenizer']:
                prompts = [
                    "总结：测试",
                    "总结以下内容：这是一个关于技术的视频",
                    "请用一句话总结：人工智能正在改变世界"
                ]
                for prompt in prompts:
                    inputs = GLOBAL_MODELS['tokenizer'](prompt, return_tensors="pt")
                    if GLOBAL_MODELS['device'] == "cuda":
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    with torch.no_grad():
                        GLOBAL_MODELS['qwen'].generate(**inputs, max_new_tokens=20, do_sample=False)

            print("[WARMUP] Extensive warmup completed")
        except Exception as e:
            print(f"[WARMUP] Warning during warmup: {e}")

    @classmethod
    async def batch_process_videos(cls, video_batch: List[Tuple[str, Dict]]) -> List[Dict]:
        """Process multiple videos in parallel with batching"""
        import torch

        results = []

        # Process in GPU batch if possible
        if GLOBAL_MODELS['device'] == "cuda" and len(video_batch) > 1:
            with torch.no_grad():
                # Batch process transcriptions
                tasks = []
                for video_url, video_info in video_batch:
                    task = cls.process_single_video(video_url, video_info)
                    tasks.append(task)

                # Execute in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential processing for CPU
            for video_url, video_info in video_batch:
                result = await cls.process_single_video(video_url, video_info)
                results.append(result)

        return results

    @classmethod
    async def process_single_video(cls, video_url: str, video_info: Dict) -> Dict:
        """Process single video with caching"""
        cache_key = cls.get_cache_key(video_url)

        # Check cache
        if cache_key in TRANSCRIPTION_CACHE:
            cached_result, timestamp = TRANSCRIPTION_CACHE[cache_key]
            if datetime.now() - timestamp < CACHE_DURATION:
                METRICS['cache_hits'] += 1
                return cached_result

        METRICS['cache_misses'] += 1
        start_time = time.time()

        try:
            # Import and use existing transcriber
            from audio_transcriber import AudioTranscriber

            transcriber = AudioTranscriber()
            transcriber.whisper_model = GLOBAL_MODELS['whisper']
            transcriber.qwen_model = GLOBAL_MODELS['qwen']
            transcriber.qwen_tokenizer = GLOBAL_MODELS['tokenizer']
            transcriber.device = GLOBAL_MODELS['device']

            result = await transcriber.transcribe_video_audio(video_url, video_info)

            # Cache result
            TRANSCRIPTION_CACHE[cache_key] = (result, datetime.now())

            # Limit cache size
            if len(TRANSCRIPTION_CACHE) > MAX_CACHE_SIZE:
                oldest = min(TRANSCRIPTION_CACHE.items(), key=lambda x: x[1][1])
                del TRANSCRIPTION_CACHE[oldest[0]]

            # Record metrics
            processing_time = time.time() - start_time
            METRICS['processing_times'].append(processing_time)

            return result

        except Exception as e:
            print(f"[ERROR] Processing failed for {video_url}: {e}")
            return {
                'transcript': video_info.get('title', ''),
                'paragraph_summary': video_info.get('title', ''),
                'sentence_subtitle': video_info.get('title', '')[:50],
                'status': 'error'
            }

    @classmethod
    def get_cache_key(cls, video_url: str) -> str:
        """Generate cache key"""
        return hashlib.md5(video_url.encode()).hexdigest()

# Create FastAPI app
app = FastAPI(
    title="Ultra-Optimized Video RSS Aggregator",
    description="Maximum performance RSS aggregator with AI transcription",
    version="3.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("[STARTUP] Ultra-Optimized Video RSS Aggregator starting...")
    print("[STARTUP] Initializing high-performance components...")

    # Start model initialization
    asyncio.create_task(UltraOptimizedTranscriber.initialize_global_models())

    # Start batch processor
    asyncio.create_task(batch_processor())

    # Load env
    from dotenv import load_dotenv
    load_dotenv()

    print("[STARTUP] Server ready at http://0.0.0.0:8000")

async def batch_processor():
    """Background batch processor for optimal throughput"""
    while True:
        try:
            batch = []
            deadline = time.time() + BATCH_TIMEOUT

            # Collect batch
            while len(batch) < BATCH_SIZE and time.time() < deadline:
                try:
                    timeout = max(0, deadline - time.time())
                    item = await asyncio.wait_for(BATCH_QUEUE.get(), timeout=timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if batch:
                # Process batch
                await UltraOptimizedTranscriber.batch_process_videos(batch)

        except Exception as e:
            print(f"[BATCH] Error in batch processor: {e}")
            await asyncio.sleep(1)

@app.get("/")
async def root():
    """Root endpoint with metrics"""
    avg_time = sum(METRICS['processing_times']) / len(METRICS['processing_times']) if METRICS['processing_times'] else 0

    return {
        "name": "Ultra-Optimized Video RSS Aggregator",
        "version": "3.0",
        "status": "running",
        "models_loaded": GLOBAL_MODELS['initialized'],
        "metrics": {
            "total_requests": METRICS['total_requests'],
            "cache_hits": METRICS['cache_hits'],
            "cache_misses": METRICS['cache_misses'],
            "cache_hit_rate": METRICS['cache_hits'] / max(1, METRICS['cache_hits'] + METRICS['cache_misses']),
            "avg_processing_time": f"{avg_time:.3f}s",
            "cache_size": len(TRANSCRIPTION_CACHE)
        }
    }

@app.get("/health")
async def health_check():
    """Health check with detailed status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "loaded": GLOBAL_MODELS['initialized'],
            "whisper": GLOBAL_MODELS['whisper'] is not None,
            "qwen": GLOBAL_MODELS['qwen'] is not None,
            "device": GLOBAL_MODELS.get('device', 'not_initialized')
        },
        "performance": {
            "cache_entries": len(TRANSCRIPTION_CACHE),
            "cache_hit_rate": f"{(METRICS['cache_hits'] / max(1, METRICS['cache_hits'] + METRICS['cache_misses']) * 100):.1f}%",
            "active_threads": threading.active_count()
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Detailed performance metrics"""
    processing_times = list(METRICS['processing_times'])

    return {
        "requests": {
            "total": METRICS['total_requests'],
            "cache_hits": METRICS['cache_hits'],
            "cache_misses": METRICS['cache_misses']
        },
        "latency": {
            "min": f"{min(processing_times):.3f}s" if processing_times else "N/A",
            "max": f"{max(processing_times):.3f}s" if processing_times else "N/A",
            "avg": f"{sum(processing_times)/len(processing_times):.3f}s" if processing_times else "N/A",
            "p50": f"{sorted(processing_times)[len(processing_times)//2]:.3f}s" if processing_times else "N/A",
            "p95": f"{sorted(processing_times)[int(len(processing_times)*0.95)]:.3f}s" if len(processing_times) > 20 else "N/A"
        },
        "cache": {
            "size": len(TRANSCRIPTION_CACHE),
            "max_size": MAX_CACHE_SIZE,
            "hit_rate": f"{(METRICS['cache_hits'] / max(1, METRICS['cache_hits'] + METRICS['cache_misses']) * 100):.1f}%"
        },
        "system": {
            "threads": threading.active_count(),
            "device": GLOBAL_MODELS.get('device', 'not_initialized')
        }
    }

@app.get("/rss/{platform}")
async def get_ultra_optimized_rss(
    platform: str,
    limit: int = Query(10, ge=1, le=50),
    personalized: bool = Query(True),
    full_transcription: bool = Query(True)
):
    """Ultra-optimized RSS endpoint"""

    METRICS['total_requests'] += 1
    start_time = time.time()

    try:
        # Get videos
        from bilibili_recommendations import BilibiliRecommendationFetcher

        if platform.lower() == "bilibili":
            async with BilibiliRecommendationFetcher() as bili:
                videos = await bili.fetch_recommendations(limit=limit)
        else:
            videos = []

        # Process with ultra-optimization
        if full_transcription and videos:
            # Create batch for processing
            video_batch = [(f"https://www.bilibili.com/video/{v.get('bvid', '')}", v) for v in videos]

            # Process in optimal batches
            results = await UltraOptimizedTranscriber.batch_process_videos(video_batch)

            # Add results
            for video, result in zip(videos, results):
                video['full_transcription'] = result

        # Generate RSS
        from xml.etree import ElementTree as ET
        rss = ET.Element("rss", version="2.0")
        channel = ET.SubElement(rss, "channel")
        ET.SubElement(channel, "title").text = f"Ultra-Fast AI Digest - {platform.title()}"

        for video in videos:
            item = ET.SubElement(channel, "item")
            ET.SubElement(item, "title").text = video.get('title', '')
            ET.SubElement(item, "link").text = f"https://www.bilibili.com/video/{video.get('bvid', '')}"

            if 'full_transcription' in video:
                trans = video['full_transcription']
                ET.SubElement(item, "description").text = f"摘要: {trans.get('paragraph_summary', '')}"

        rss_xml = ET.tostring(rss, encoding='unicode')

        # Record latency
        total_time = time.time() - start_time
        METRICS['processing_times'].append(total_time)

        return Response(
            content=rss_xml,
            media_type="application/xml",
            headers={"X-Processing-Time": f"{total_time:.3f}s"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "ultra_optimized_backend:app",
        host="0.0.0.0",
        port=8000,  # Different port for testing
        workers=1,
        log_level="info"
    )
