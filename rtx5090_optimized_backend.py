#!/usr/bin/env python3
"""
RTX 5090 Ultra-Optimized Backend with REAL Bilibili Data Integration
Maximizes GPU utilization for sub-second processing per video using authentic content
"""

import os
import sys
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
import whisper
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import xml.etree.ElementTree as ET
from datetime import datetime
import uvicorn
import yt_dlp
import tempfile
import numpy as np
import librosa

# Set UTF-8 encoding
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file for Bilibili authentication"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = value.strip()
        print("[OK] Loaded credentials from .env file")
        print("   - Bilibili authentication configured")
    else:
        print("[WARNING] .env file not found - using public APIs only")

load_env_file()

# Real video data cache
REAL_VIDEO_CACHE = {
    "bilibili": [],
    "douyin": [],
    "kuaishou": []
}

# GPU Configuration for RTX 5090
def setup_rtx5090_optimization():
    """Configure maximum RTX 5090 utilization"""
    if torch.cuda.is_available():
        # Enable maximum GPU utilization
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set memory allocation strategy
        torch.cuda.empty_cache()
        torch.cuda.memory.set_per_process_memory_fraction(0.95)  # Use 95% of 32GB

        # Enable mixed precision for maximum speed
        torch.set_default_dtype(torch.float16)

        device = torch.device("cuda:0")
        print(f"🚀 RTX 5090 Optimization: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        return device
    else:
        print("❌ CUDA not available")
        return torch.device("cpu")

class RTX5090TranscriptionEngine:
    """Ultra-fast transcription engine for RTX 5090 with real Bilibili data"""

    def __init__(self):
        self.device = setup_rtx5090_optimization()
        self.whisper_model = None
        self.qwen_model = None
        self.qwen_tokenizer = None

        # Batch processing settings
        self.max_batch_size = 8  # Process 8 videos simultaneously
        self.executor = ThreadPoolExecutor(max_workers=16)  # CPU parallel processing

        # Real data extractor for authentic Bilibili content
        self.real_data_cache = REAL_VIDEO_CACHE

    async def initialize_models(self):
        """Load models with maximum RTX 5090 optimization"""
        print("[RTX5090] Loading Whisper with maximum GPU acceleration...")
        start_time = time.time()

        # Whisper: Use largest model with maximum GPU utilization
        self.whisper_model = whisper.load_model(
            "large-v3",
            device=self.device,
            download_root=None
        )
        # Enable mixed precision
        if hasattr(self.whisper_model, 'encoder'):
            self.whisper_model.encoder = self.whisper_model.encoder.to(dtype=torch.float16)
        if hasattr(self.whisper_model, 'decoder'):
            self.whisper_model.decoder = self.whisper_model.decoder.to(dtype=torch.float16)

        whisper_time = time.time() - start_time
        print(f"✅ Whisper loaded in {whisper_time:.2f}s")

        # Qwen2.5: Maximum quantization for speed
        print("[RTX5090] Loading Qwen2.5-7B with 4-bit quantization...")
        qwen_start = time.time()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=True,
            low_cpu_mem_usage=True
        )

        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            padding_side="left"
        )
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token

        qwen_time = time.time() - qwen_start
        print(f"✅ Qwen2.5-7B loaded in {qwen_time:.2f}s")

        # Warmup both models
        await self.warmup_models()

        total_time = time.time() - start_time
        print(f"🚀 RTX 5090 models ready in {total_time:.2f}s total")

    async def warmup_models(self):
        """Warmup models for maximum performance"""
        print("[RTX5090] Warming up models...")

        # Warmup Whisper
        dummy_audio = torch.randn(16000, dtype=torch.float32)  # 1 second audio
        with torch.cuda.amp.autocast():
            _ = self.whisper_model.transcribe(dummy_audio.numpy(), fp16=True)

        # Warmup Qwen
        dummy_text = "测试"
        inputs = self.qwen_tokenizer(dummy_text, return_tensors="pt", padding=True).to(self.device)
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                _ = self.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.qwen_tokenizer.eos_token_id
                )

        print("✅ Models warmed up")

    async def download_audio_ultra_fast(self, video_url: str) -> str:
        """Download audio from video URL using yt-dlp"""
        try:
            # Create temporary file for audio
            temp_file = tempfile.mktemp(suffix='.wav')

            # yt-dlp options for fast audio extraction
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'outtmpl': temp_file.replace('.wav', ''),
                'quiet': True,
                'no_warnings': True,
            }

            # Download with executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: yt_dlp.YoutubeDL(ydl_opts).download([video_url])
            )

            return temp_file

        except Exception as e:
            print(f"[ERROR] Audio download failed: {e}")
            return None

    async def transcribe_audio_ultra_fast(self, audio_file: str) -> str:
        """Transcribe audio with maximum GPU utilization"""
        try:
            # Load and preprocess audio on CPU with executor
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(
                self.executor,
                lambda: librosa.load(audio_file, sr=16000)[0]
            )

            # Transcribe with Whisper using GPU
            with torch.cuda.amp.autocast():
                result = self.whisper_model.transcribe(
                    audio_data,
                    fp16=True,
                    language='zh',
                    task='transcribe',
                    beam_size=1,
                    temperature=0.0,
                    condition_on_previous_text=False
                )

            return result['text'].strip()

        except Exception as e:
            print(f"[ERROR] Transcription failed: {e}")
            return "转录失败"

    async def get_real_bilibili_videos(self, limit: int = 5) -> List[Dict]:
        """Fetch real Bilibili videos using the data extractor"""
        try:
            from real_data_extractor import RealDataExtractor
            print(f"[RTX5090] Fetching {limit} real Bilibili videos...")

            async with RealDataExtractor() as extractor:
                real_videos = await extractor.extract_all_real_data()
                bilibili_videos = real_videos.get('bilibili', [])[:limit]

                if bilibili_videos:
                    self.real_data_cache['bilibili'] = bilibili_videos
                    print(f"✅ Fetched {len(bilibili_videos)} real Bilibili videos")
                    return bilibili_videos
                else:
                    print("[WARNING] No real Bilibili videos found, using fallback")
                    return []

        except Exception as e:
            print(f"[ERROR] Failed to fetch real videos: {e}")
            return []

    async def batch_transcribe_real_videos(self, limit: int = 5) -> List[Dict]:
        """Process real Bilibili videos with RTX 5090 acceleration"""
        # First get real videos
        real_videos = await self.get_real_bilibili_videos(limit)

        if not real_videos:
            print("[RTX5090] No real videos available, returning empty results")
            return []

        print(f"[RTX5090] Processing {len(real_videos)} real videos with AI transcription...")
        start_time = time.time()

        # Process real videos with AI enhancement
        all_results = []
        for video in real_videos:
            try:
                # Extract video URL for processing
                video_url = video.get('url', '')
                if not video_url:
                    continue

                # Process with RTX 5090 acceleration
                enhanced_video = await self.process_real_video_ultra_fast(video, video_url)
                all_results.append(enhanced_video)

            except Exception as e:
                print(f"[ERROR] Failed to process video {video.get('title', 'Unknown')}: {e}")
                # Still include the original video data even if AI processing fails
                all_results.append(video)

        total_time = time.time() - start_time
        avg_time = total_time / len(real_videos) if real_videos else 0
        print(f"✅ RTX 5090 processed {len(real_videos)} real videos in {total_time:.2f}s (avg: {avg_time:.2f}s/video)")

        return all_results

    async def process_real_video_ultra_fast(self, video_data: Dict, video_url: str) -> Dict:
        """Process real video data with RTX 5090 AI enhancement"""
        processing_start = time.time()

        try:
            # Start with real video metadata
            enhanced_video = video_data.copy()

            # Add AI transcription if video URL is available
            if video_url and 'bilibili.com' in video_url:
                try:
                    # Download audio using yt-dlp (CPU intensive)
                    audio_file = await self.download_audio_ultra_fast(video_url)

                    if audio_file:
                        # Transcribe with Whisper (GPU intensive)
                        transcript = await self.transcribe_audio_ultra_fast(audio_file)

                        # Generate AI summary with Qwen (GPU intensive)
                        ai_summary = await self.generate_summary_ultra_fast(transcript)

                        # Enhanced with AI analysis
                        enhanced_video.update({
                            'ai_transcript': transcript,
                            'ai_summary': ai_summary,
                            'ai_enhanced': True
                        })

                        # Clean up temporary file
                        try:
                            os.unlink(audio_file)
                        except:
                            pass

                except Exception as ai_error:
                    print(f"[WARNING] AI processing failed for {video_data.get('title', 'Unknown')}: {ai_error}")
                    enhanced_video['ai_enhanced'] = False
                    enhanced_video['ai_error'] = str(ai_error)

            # Add processing metadata
            processing_time = time.time() - processing_start
            enhanced_video['rtx5090_processing_time'] = processing_time
            enhanced_video['processed_at'] = datetime.now().isoformat()

            return enhanced_video

        except Exception as e:
            print(f"[ERROR] Video processing failed: {e}")
            # Return original video data even if enhancement fails
            video_data['rtx5090_processing_error'] = str(e)
            return video_data

    async def generate_summary_ultra_fast(self, transcript: str) -> str:
        """Ultra-fast summary generation"""
        if not transcript or len(transcript) < 10:
            return "内容较短，无需总结"

        try:
            # Optimized prompt for speed
            prompt = f"请用一句话总结：{transcript[:200]}..."

            inputs = self.qwen_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Ultra-fast generation
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.qwen_model.generate(
                        inputs.input_ids,
                        max_new_tokens=50,  # Short summaries for speed
                        do_sample=False,
                        temperature=0.1,
                        pad_token_id=self.qwen_tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        use_cache=True
                    )

            summary = self.qwen_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            return summary if summary else "AI总结生成中..."

        except Exception as e:
            print(f"[ERROR] Summary generation failed: {e}")
            return "总结生成失败"

# Global transcription engine
transcription_engine = RTX5090TranscriptionEngine()

# FastAPI App
app = FastAPI(title="RTX 5090 Ultra-Optimized Video RSS with Real Data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize RTX 5090 optimization"""
    print("🚀 Starting RTX 5090 Ultra-Optimized Backend with Real Bilibili Data...")
    await transcription_engine.initialize_models()
    print("✅ RTX 5090 Backend ready!")

@app.get("/health")
async def health_check():
    """Health check with GPU status"""
    gpu_info = {
        "status": "healthy",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
        "gpu_memory_used": f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A",
        "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else "N/A",
        "real_data_integration": "enabled",
        "bilibili_auth": "configured" if os.environ.get("BILIBILI_SESSDATA") else "public_only"
    }
    return gpu_info

@app.get("/rss/bilibili", response_class=PlainTextResponse)
async def get_rtx5090_rss_feed(limit: int = 5):
    """RTX 5090 optimized RSS feed with REAL Bilibili data"""
    try:
        print(f"[RTX5090] Generating RSS feed with {limit} REAL Bilibili videos...")
        start_time = time.time()

        # Process REAL Bilibili videos with RTX 5090 AI enhancement
        videos = await transcription_engine.batch_transcribe_real_videos(limit)

        if not videos:
            print("[WARNING] No real videos processed, generating empty RSS feed")
            videos = []

        # Generate RSS with real data
        rss = ET.Element("rss", version="2.0")
        rss.set("xmlns:content", "http://purl.org/rss/1.0/modules/content/")

        channel = ET.SubElement(rss, "channel")
        ET.SubElement(channel, "title").text = "Ultra-Fast AI Video Feed"
        ET.SubElement(channel, "description").text = "RTX 5090 GPU-accelerated real Bilibili video transcription"
        ET.SubElement(channel, "link").text = "http://111.186.3.124:8000"
        ET.SubElement(channel, "lastBuildDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

        for video in videos:
            item = ET.SubElement(channel, "item")

            # Use real video data
            title = video.get('title', 'No Title')
            author = video.get('author', 'Unknown Author')
            video_url = video.get('url', '')

            # Create rich description with AI enhancement
            description_parts = []

            # Add AI summary if available
            if video.get('ai_enhanced') and video.get('ai_summary'):
                description_parts.append(f"🤖 AI智能摘要：{video.get('ai_summary', '')}")

            # Add original description
            if video.get('description'):
                description_parts.append(f"📝 原描述：{video.get('description', '')[:100]}...")

            # Add AI transcript excerpt if available
            if video.get('ai_transcript'):
                description_parts.append(f"📄 AI转录节选：{video.get('ai_transcript', '')[:100]}...")

            # Add video stats
            stats = []
            if video.get('view_count'):
                stats.append(f"播放：{video.get('view_count')}")
            if video.get('like_count'):
                stats.append(f"点赞：{video.get('like_count')}")
            if video.get('duration'):
                stats.append(f"时长：{video.get('duration')}")

            if stats:
                description_parts.append(f"📊 数据：{' | '.join(stats)}")

            # Add processing info
            if video.get('rtx5090_processing_time'):
                processing_time = video.get('rtx5090_processing_time', 0)
                description_parts.append(f"⚡ RTX 5090处理时间：{processing_time:.2f}s")

            description = "\n\n".join(description_parts)

            ET.SubElement(item, "title").text = title
            ET.SubElement(item, "link").text = video_url
            ET.SubElement(item, "author").text = author
            ET.SubElement(item, "description").text = description

            # Use real upload date if available
            pub_date = video.get('upload_date')
            if pub_date:
                try:
                    # Convert YYYY-MM-DD to RSS format
                    date_obj = datetime.strptime(pub_date, '%Y-%m-%d')
                    pub_date_rss = date_obj.strftime("%a, %d %b %Y %H:%M:%S GMT")
                except:
                    pub_date_rss = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
            else:
                pub_date_rss = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

            ET.SubElement(item, "pubDate").text = pub_date_rss

        total_time = time.time() - start_time
        processed_count = len([v for v in videos if v.get('ai_enhanced')])
        print(f"✅ RTX 5090 RSS generated in {total_time:.2f}s for {len(videos)} videos ({processed_count} AI-enhanced)")

        return ET.tostring(rss, encoding="unicode")

    except Exception as e:
        print(f"[ERROR] RSS generation failed: {e}")
        import traceback
        traceback.print_exc()
        return f"<?xml version='1.0'?><rss version='2.0'><channel><title>Error</title><description>RTX 5090 Backend Error: {e}</description></channel></rss>"

if __name__ == "__main__":
    print("🚀 Starting RTX 5090 Ultra-Optimized Backend with Real Bilibili Data...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9001,  # Use port 9001 for RTX 5090 version
        log_level="info",
        access_log=False
    )