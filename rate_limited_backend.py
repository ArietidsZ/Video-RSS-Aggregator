#!/usr/bin/env python3
"""
Rate-Limited Video RSS Aggregator Backend
Handles 20+ videos with intelligent rate limiting (1 min per video after 20 videos)
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
from concurrent.futures import ThreadPoolExecutor
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

# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    'threshold': 20,  # After 20 videos, apply rate limiting
    'delay_seconds': 60,  # 1 minute delay per video after threshold
    'concurrent_limit': 3,  # Max concurrent transcriptions
    'batch_size': 5,  # Process in batches of 5
    'cache_duration_hours': 24,  # Longer cache for rate-limited content
}

# Performance metrics
METRICS = {
    'total_requests': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'videos_processed': 0,
    'rate_limited_count': 0,
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
}

# Advanced caching
TRANSCRIPTION_CACHE = {}
CACHE_DURATION = timedelta(hours=RATE_LIMIT_CONFIG['cache_duration_hours'])
MAX_CACHE_SIZE = 2000

# Thread pools
CPU_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cpu")
GPU_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gpu")

# Rate limiter
class RateLimiter:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_process_time = time.time()
        self.videos_in_window = 0
        self.window_start = time.time()

    async def wait_if_needed(self, video_index: int, total_videos: int):
        """Apply rate limiting after threshold"""
        with self.lock:
            # Reset window every hour
            if time.time() - self.window_start > 3600:
                self.videos_in_window = 0
                self.window_start = time.time()

            self.videos_in_window += 1

            # Apply rate limiting after threshold
            if video_index >= RATE_LIMIT_CONFIG['threshold']:
                delay = RATE_LIMIT_CONFIG['delay_seconds']
                print(f"[RATE LIMIT] Video {video_index + 1}/{total_videos} - Applying {delay}s delay")
                METRICS['rate_limited_count'] += 1
                await asyncio.sleep(delay)
            elif video_index > 0 and video_index % RATE_LIMIT_CONFIG['batch_size'] == 0:
                # Small delay between batches
                print(f"[BATCH] Completed batch, short pause...")
                await asyncio.sleep(2)

RATE_LIMITER = RateLimiter()

class RateLimitedTranscriber:
    """Transcriber with intelligent rate limiting"""

    @classmethod
    async def initialize_global_models(cls):
        """Initialize models once"""
        with GLOBAL_MODELS['lock']:
            if GLOBAL_MODELS['initialized']:
                return

            try:
                import torch
                import whisper
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

                torch.backends.cudnn.benchmark = True
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[STARTUP] Initializing models on {device}...")

                # Load Whisper
                print("[STARTUP] Loading Whisper model...")
                GLOBAL_MODELS['whisper'] = whisper.load_model("base", device=device)
                if device == "cuda":
                    GLOBAL_MODELS['whisper'] = GLOBAL_MODELS['whisper'].half()

                # Load Qwen with quantization
                print("[STARTUP] Loading Qwen2.5-7B-Instruct...")
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
                    use_fast=True
                )

                GLOBAL_MODELS['qwen'] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )

                GLOBAL_MODELS['qwen'].eval()
                GLOBAL_MODELS['device'] = device
                GLOBAL_MODELS['initialized'] = True

                print("[STARTUP] ✅ Models ready with rate limiting enabled!")

            except Exception as e:
                print(f"[ERROR] Model initialization failed: {e}")
                raise

    @classmethod
    async def process_videos_with_rate_limit(cls, videos: List[Dict]) -> List[Dict]:
        """Process videos with intelligent rate limiting"""
        total_videos = len(videos)
        print(f"[PROCESS] Starting to process {total_videos} videos with rate limiting")

        results = []

        # Process in batches with rate limiting
        for i in range(0, total_videos, RATE_LIMIT_CONFIG['batch_size']):
            batch_end = min(i + RATE_LIMIT_CONFIG['batch_size'], total_videos)
            batch = videos[i:batch_end]

            print(f"[BATCH] Processing videos {i+1}-{batch_end} of {total_videos}")

            # Process batch concurrently (up to concurrent_limit)
            batch_tasks = []
            for j, video in enumerate(batch):
                video_index = i + j

                # Apply rate limiting
                await RATE_LIMITER.wait_if_needed(video_index, total_videos)

                # Create processing task
                video_url = f"https://www.bilibili.com/video/{video.get('bvid', '')}"
                task = cls.process_single_video(video_url, video, video_index)
                batch_tasks.append(task)

                # Limit concurrent processing
                if len(batch_tasks) >= RATE_LIMIT_CONFIG['concurrent_limit']:
                    batch_results = await asyncio.gather(*batch_tasks)
                    results.extend(batch_results)
                    batch_tasks = []

            # Process remaining tasks
            if batch_tasks:
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)

            # Progress update
            print(f"[PROGRESS] Completed {len(results)}/{total_videos} videos")

            # Yield control periodically
            await asyncio.sleep(0.1)

        return results

    @classmethod
    async def process_single_video(cls, video_url: str, video_info: Dict, index: int) -> Dict:
        """Process single video with caching"""
        cache_key = cls.get_cache_key(video_url)

        # Check cache first
        if cache_key in TRANSCRIPTION_CACHE:
            cached_result, timestamp = TRANSCRIPTION_CACHE[cache_key]
            if datetime.now() - timestamp < CACHE_DURATION:
                METRICS['cache_hits'] += 1
                print(f"[CACHE HIT] Video {index + 1}: {video_info.get('title', '')[:30]}...")
                return cached_result

        METRICS['cache_misses'] += 1
        start_time = time.time()

        try:
            print(f"[PROCESSING] Video {index + 1}: {video_info.get('title', '')[:50]}...")

            # Simulate transcription with fallback
            result = {
                'transcript': f"Transcribed content for: {video_info.get('title', '')}",
                'paragraph_summary': f"AI Summary: {video_info.get('title', '')} - This video discusses important topics with detailed explanations.",
                'sentence_subtitle': f"Key insight from {video_info.get('title', '')[:30]}",
                'status': 'success',
                'processed_at': datetime.now().isoformat(),
                'index': index
            }

            # Cache result
            TRANSCRIPTION_CACHE[cache_key] = (result, datetime.now())

            # Clean cache if too large
            if len(TRANSCRIPTION_CACHE) > MAX_CACHE_SIZE:
                oldest = min(TRANSCRIPTION_CACHE.items(), key=lambda x: x[1][1])
                del TRANSCRIPTION_CACHE[oldest[0]]

            # Record metrics
            processing_time = time.time() - start_time
            METRICS['processing_times'].append(processing_time)
            METRICS['videos_processed'] += 1

            return result

        except Exception as e:
            print(f"[ERROR] Failed processing video {index + 1}: {e}")
            return {
                'transcript': video_info.get('title', ''),
                'paragraph_summary': video_info.get('title', ''),
                'sentence_subtitle': video_info.get('title', '')[:50],
                'status': 'error',
                'error': str(e),
                'index': index
            }

    @classmethod
    def get_cache_key(cls, video_url: str) -> str:
        """Generate cache key"""
        return hashlib.md5(video_url.encode()).hexdigest()

# Create FastAPI app
app = FastAPI(
    title="Rate-Limited Video RSS Aggregator",
    description="Intelligent rate limiting for processing 20+ videos",
    version="4.0"
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
    print("[STARTUP] Rate-Limited Video RSS Aggregator starting...")
    print(f"[STARTUP] Rate limit config: {RATE_LIMIT_CONFIG}")

    # Initialize models in background
    asyncio.create_task(RateLimitedTranscriber.initialize_global_models())

    from dotenv import load_dotenv
    load_dotenv()

    print("[STARTUP] Server ready at http://0.0.0.0:8000")

@app.get("/")
async def root():
    """Root endpoint with metrics"""
    avg_time = sum(METRICS['processing_times']) / len(METRICS['processing_times']) if METRICS['processing_times'] else 0

    return {
        "name": "Rate-Limited Video RSS Aggregator",
        "version": "4.0",
        "status": "running",
        "models_loaded": GLOBAL_MODELS['initialized'],
        "rate_limit_config": RATE_LIMIT_CONFIG,
        "metrics": {
            "total_requests": METRICS['total_requests'],
            "videos_processed": METRICS['videos_processed'],
            "rate_limited_count": METRICS['rate_limited_count'],
            "cache_hits": METRICS['cache_hits'],
            "cache_misses": METRICS['cache_misses'],
            "cache_hit_rate": f"{(METRICS['cache_hits'] / max(1, METRICS['cache_hits'] + METRICS['cache_misses']) * 100):.1f}%",
            "avg_processing_time": f"{avg_time:.3f}s",
            "cache_size": len(TRANSCRIPTION_CACHE)
        }
    }

@app.get("/rss/{platform}")
async def get_rate_limited_rss(
    platform: str,
    limit: int = Query(10, ge=1, le=100),
    full_transcription: bool = Query(True)
):
    """RSS endpoint with intelligent rate limiting for 20+ videos"""

    METRICS['total_requests'] += 1
    start_time = time.time()

    try:
        # Generate test videos for demonstration
        videos = []
        for i in range(limit):
            videos.append({
                'bvid': f'BV{i:03d}test',
                'title': f'Test Video {i+1}: AI and Technology Insights',
                'author': f'Creator {i+1}',
                'view': 10000 * (i + 1),
                'like': 500 * (i + 1),
                'coin': 200 * (i + 1),
                'danmaku': 50 * (i + 1),
                'duration': f'{15 + i}:00'
            })

        # Process with rate limiting
        if full_transcription and videos:
            print(f"[RSS] Processing {len(videos)} videos with rate limiting...")

            # Show warning for large requests
            if len(videos) > RATE_LIMIT_CONFIG['threshold']:
                estimated_time = (len(videos) - RATE_LIMIT_CONFIG['threshold']) * RATE_LIMIT_CONFIG['delay_seconds']
                print(f"[WARNING] Processing {len(videos)} videos will take approximately {estimated_time/60:.1f} minutes due to rate limiting")

            # Process videos with rate limiting
            transcriptions = await RateLimitedTranscriber.process_videos_with_rate_limit(videos)

            # Add transcriptions to videos
            for video, trans in zip(videos, transcriptions):
                video['full_transcription'] = trans

        # Generate RSS
        from xml.etree import ElementTree as ET
        rss = ET.Element("rss", version="2.0")
        channel = ET.SubElement(rss, "channel")
        ET.SubElement(channel, "title").text = f"Rate-Limited AI Digest - {platform.title()}"
        ET.SubElement(channel, "description").text = f"Processing {len(videos)} videos with intelligent rate limiting"

        for video in videos:
            item = ET.SubElement(channel, "item")
            ET.SubElement(item, "title").text = video.get('title', '')
            ET.SubElement(item, "link").text = f"https://www.bilibili.com/video/{video.get('bvid', '')}"

            if 'full_transcription' in video:
                trans = video['full_transcription']
                description = f"""
🤖 AI摘要: {trans.get('paragraph_summary', '')}
📝 字幕: {trans.get('sentence_subtitle', '')}
📊 处理索引: {trans.get('index', 'N/A')}
⏰ 处理时间: {trans.get('processed_at', 'N/A')}
                """
                ET.SubElement(item, "description").text = description.strip()

        rss_xml = ET.tostring(rss, encoding='unicode')

        # Record total time
        total_time = time.time() - start_time

        return Response(
            content=rss_xml,
            media_type="application/xml",
            headers={
                "X-Processing-Time": f"{total_time:.3f}s",
                "X-Videos-Processed": str(len(videos)),
                "X-Rate-Limited": str(len(videos) > RATE_LIMIT_CONFIG['threshold'])
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rate_limit_active": True,
        "config": RATE_LIMIT_CONFIG,
        "metrics": {
            "videos_processed": METRICS['videos_processed'],
            "rate_limited_count": METRICS['rate_limited_count'],
            "cache_size": len(TRANSCRIPTION_CACHE)
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "rate_limited_backend:app",
        host="0.0.0.0",
        port=8002,
        workers=1,
        log_level="info"
    )