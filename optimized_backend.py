#!/usr/bin/env python3
"""
Optimized Video RSS Aggregator Backend
High-performance API with preloaded models and caching
"""

import asyncio
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import hashlib
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Set UTF-8 encoding
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import uvicorn

# Global model instances for persistence
GLOBAL_MODELS = {
    'whisper': None,
    'qwen': None,
    'tokenizer': None,
    'device': None,
    'initialized': False,
    'lock': threading.Lock()
}

# Cache for transcription results
TRANSCRIPTION_CACHE = {}
CACHE_DURATION = timedelta(hours=6)

# Thread pool for parallel processing
EXECUTOR = ThreadPoolExecutor(max_workers=4)

class OptimizedTranscriber:
    """Optimized transcriber with model persistence and caching"""

    @classmethod
    async def initialize_global_models(cls):
        """Initialize models once at startup"""
        with GLOBAL_MODELS['lock']:
            if GLOBAL_MODELS['initialized']:
                return

            try:
                import torch
                import whisper
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[STARTUP] Initializing models on {device}...")

                # Load Whisper
                print("[STARTUP] Loading Whisper model...")
                GLOBAL_MODELS['whisper'] = whisper.load_model("base", device=device)

                # Load Qwen with 4-bit quantization
                print("[STARTUP] Loading Qwen2.5-7B-Instruct (4-bit)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

                model_name = "Qwen/Qwen2.5-7B-Instruct"
                GLOBAL_MODELS['tokenizer'] = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                GLOBAL_MODELS['qwen'] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )

                GLOBAL_MODELS['device'] = device
                GLOBAL_MODELS['initialized'] = True

                # Warmup models
                print("[STARTUP] Warming up models...")
                await cls.warmup_models()

                print("[STARTUP] ✅ Models ready for high-performance inference!")

            except Exception as e:
                print(f"[ERROR] Model initialization failed: {e}")
                GLOBAL_MODELS['initialized'] = False
                raise

    @classmethod
    async def warmup_models(cls):
        """Warmup models with sample data"""
        try:
            # Warmup Whisper
            if GLOBAL_MODELS['whisper']:
                import numpy as np
                sample_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                GLOBAL_MODELS['whisper'].transcribe(sample_audio, language='zh')

            # Warmup Qwen
            if GLOBAL_MODELS['qwen'] and GLOBAL_MODELS['tokenizer']:
                test_prompt = "总结：这是一个测试视频。"
                inputs = GLOBAL_MODELS['tokenizer'](test_prompt, return_tensors="pt")
                if GLOBAL_MODELS['device'] == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                GLOBAL_MODELS['qwen'].generate(**inputs, max_new_tokens=10)

            print("[WARMUP] Models warmed up successfully")
        except Exception as e:
            print(f"[WARMUP] Warning: {e}")

    @classmethod
    def get_cache_key(cls, video_url: str, operation: str) -> str:
        """Generate cache key for transcription results"""
        return hashlib.md5(f"{video_url}:{operation}".encode()).hexdigest()

    @classmethod
    async def transcribe_with_cache(cls, video_url: str, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe with caching"""
        cache_key = cls.get_cache_key(video_url, "transcribe")

        # Check cache
        if cache_key in TRANSCRIPTION_CACHE:
            cached_result, timestamp = TRANSCRIPTION_CACHE[cache_key]
            if datetime.now() - timestamp < CACHE_DURATION:
                print(f"[CACHE] Hit for {video_url[:50]}...")
                return cached_result

        # Process if not cached
        result = await cls.process_video(video_url, video_info)

        # Store in cache
        TRANSCRIPTION_CACHE[cache_key] = (result, datetime.now())

        # Clean old cache entries
        cls.clean_cache()

        return result

    @classmethod
    def clean_cache(cls):
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in TRANSCRIPTION_CACHE.items()
            if now - timestamp > CACHE_DURATION
        ]
        for key in expired_keys:
            del TRANSCRIPTION_CACHE[key]

    @classmethod
    async def process_video(cls, video_url: str, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process video with global models"""
        if not GLOBAL_MODELS['initialized']:
            await cls.initialize_global_models()

        try:
            # Import required modules
            from audio_transcriber import AudioTranscriber

            # Create transcriber instance that uses global models
            transcriber = AudioTranscriber()
            transcriber.whisper_model = GLOBAL_MODELS['whisper']
            transcriber.qwen_model = GLOBAL_MODELS['qwen']
            transcriber.qwen_tokenizer = GLOBAL_MODELS['tokenizer']
            transcriber.device = GLOBAL_MODELS['device']

            # Process video
            result = await transcriber.transcribe_video_audio(video_url, video_info)
            return result

        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            return {
                'transcript': video_info.get('title', ''),
                'paragraph_summary': video_info.get('title', ''),
                'sentence_subtitle': video_info.get('title', '')[:50],
                'status': 'error',
                'error': str(e)
            }

# Create FastAPI app
app = FastAPI(
    title="Optimized Video RSS Aggregator",
    description="High-performance RSS aggregator with AI transcription",
    version="2.0"
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
    """Initialize models on startup"""
    print("[STARTUP] Optimized Video RSS Aggregator starting...")
    print("[STARTUP] Preloading AI models for maximum performance...")

    # Initialize models in background
    asyncio.create_task(OptimizedTranscriber.initialize_global_models())

    # Load credentials
    from dotenv import load_dotenv
    load_dotenv()

    if os.getenv('SESSDATA'):
        print("[STARTUP] ✅ Bilibili credentials loaded")

    print("[STARTUP] Server ready at http://0.0.0.0:8000")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Optimized Video RSS Aggregator",
        "version": "2.0",
        "status": "running",
        "models_loaded": GLOBAL_MODELS['initialized'],
        "cache_size": len(TRANSCRIPTION_CACHE),
        "endpoints": {
            "health": "/health",
            "rss": "/rss/{platform}",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check with model status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": GLOBAL_MODELS['initialized'],
        "whisper_ready": GLOBAL_MODELS['whisper'] is not None,
        "qwen_ready": GLOBAL_MODELS['qwen'] is not None,
        "cache_entries": len(TRANSCRIPTION_CACHE),
        "device": GLOBAL_MODELS.get('device', 'not_initialized')
    }

@app.get("/rss/{platform}")
async def get_optimized_rss(
    platform: str,
    limit: int = Query(10, ge=1, le=50),
    personalized: bool = Query(True),
    full_transcription: bool = Query(True),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Optimized RSS endpoint with parallel processing"""

    # Import required modules
    from bilibili_recommendations import BilibiliRecommendationFetcher

    try:
        # Get videos
        if platform.lower() == "bilibili":
            bili = BilibiliRecommendationFetcher()
            videos = await bili.fetch_recommendations(limit=limit)
        else:
            videos = []

        # Process transcriptions in parallel if requested
        if full_transcription and videos:
            print(f"[RSS] Processing {len(videos)} videos with AI transcription...")

            # Process videos in parallel
            tasks = []
            for video in videos:
                video_url = f"https://www.bilibili.com/video/{video.get('bvid', '')}"
                task = OptimizedTranscriber.transcribe_with_cache(video_url, video)
                tasks.append(task)

            # Wait for all transcriptions
            results = await asyncio.gather(*tasks)

            # Add results to videos
            for video, result in zip(videos, results):
                video['full_transcription'] = result

        # Generate RSS XML
        rss_xml = generate_rss_xml(videos, platform, full_transcription)

        # Schedule cache cleanup in background
        background_tasks.add_task(OptimizedTranscriber.clean_cache)

        return Response(content=rss_xml, media_type="application/xml")

    except Exception as e:
        print(f"[ERROR] RSS generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_rss_xml(videos: List[Dict], platform: str, include_transcription: bool) -> str:
    """Generate RSS XML with transcription data"""
    from xml.etree import ElementTree as ET
    from datetime import datetime

    # Create RSS root
    rss = ET.Element("rss", version="2.0")
    rss.set("xmlns:content", "http://purl.org/rss/1.0/modules/content/")

    channel = ET.SubElement(rss, "channel")

    # Channel metadata
    ET.SubElement(channel, "title").text = f"AI Video Digest - {platform.title()}"
    ET.SubElement(channel, "description").text = "High-performance AI-powered video content aggregator"
    ET.SubElement(channel, "link").text = "http://localhost:8000"
    ET.SubElement(channel, "lastBuildDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Add video items
    for video in videos:
        item = ET.SubElement(channel, "item")

        ET.SubElement(item, "title").text = video.get('title', 'Untitled')
        ET.SubElement(item, "link").text = f"https://www.bilibili.com/video/{video.get('bvid', '')}"

        # Add transcription content if available
        if include_transcription and 'full_transcription' in video:
            trans = video['full_transcription']

            description = f"""
🤖 AI摘要：{trans.get('paragraph_summary', '')}
📝 字幕：{trans.get('sentence_subtitle', '')}
            """
            ET.SubElement(item, "description").text = description

            # Add full content
            content = f"""
<p><strong>段落摘要：</strong>{trans.get('paragraph_summary', '')}</p>
<p><strong>句子字幕：</strong>{trans.get('sentence_subtitle', '')}</p>
<p><strong>转录状态：</strong>{trans.get('status', 'unknown')}</p>
            """
            content_elem = ET.SubElement(item, "{http://purl.org/rss/1.0/modules/content/}encoded")
            content_elem.text = content

    return ET.tostring(rss, encoding='unicode')

if __name__ == "__main__":
    uvicorn.run(
        "optimized_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production
        workers=1,  # Single worker to maintain model state
        log_level="info"
    )
