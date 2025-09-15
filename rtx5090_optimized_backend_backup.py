#!/usr/bin/env python3
"""
RTX 5090 Ultra-Optimized Backend
Maximizes GPU utilization for sub-second processing per video
"""

import os
import sys
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
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
    """Ultra-fast transcription engine for RTX 5090"""

    def __init__(self):
        self.device = setup_rtx5090_optimization()
        self.whisper_model = None
        self.qwen_model = None
        self.qwen_tokenizer = None

        # Batch processing settings
        self.max_batch_size = 8  # Process 8 videos simultaneously
        self.executor = ThreadPoolExecutor(max_workers=16)  # CPU parallel processing

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

    async def batch_transcribe(self, video_urls: List[str]) -> List[Dict]:
        """Process multiple videos in parallel batches"""
        if not video_urls:
            return []

        print(f"[RTX5090] Batch processing {len(video_urls)} videos...")
        start_time = time.time()

        # Split into batches
        batches = [video_urls[i:i+self.max_batch_size]
                  for i in range(0, len(video_urls), self.max_batch_size)]

        all_results = []
        for batch in batches:
            # Process batch in parallel
            tasks = [self.process_single_video_ultra_fast(url) for url in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter successful results
            valid_results = [r for r in batch_results if not isinstance(r, Exception)]
            all_results.extend(valid_results)

        total_time = time.time() - start_time
        avg_time = total_time / len(video_urls) if video_urls else 0
        print(f"✅ RTX 5090 processed {len(video_urls)} videos in {total_time:.2f}s (avg: {avg_time:.2f}s/video)")

        return all_results

    async def process_single_video_ultra_fast(self, video_url: str) -> Dict:
        """Ultra-fast single video processing"""
        try:
            # Real video processing with RTX 5090 acceleration
            processing_start = time.time()

            # Extract video ID from URL
            video_id = video_url.split('/')[-1] if '/' in video_url else video_url[-10:]

            # Download audio using yt-dlp (CPU intensive)
            audio_file = await self.download_audio_ultra_fast(video_url)

            if not audio_file:
                raise Exception(f"Failed to download audio from {video_url}")

            # Transcribe with Whisper (GPU intensive)
            transcript = await self.transcribe_audio_ultra_fast(audio_file)

            # Generate summary with Qwen (GPU intensive)
            summary = await self.generate_summary_ultra_fast(transcript)

            # Clean up temporary file
            try:
                os.unlink(audio_file)
            except:
                pass

            processing_time = time.time() - processing_start

            return {
                'url': video_url,
                'title': f'Video {video_id}',
                'author': 'Creator',
                'transcript': transcript,
                'summary': summary,
                'processing_time': processing_time
            }

        except Exception as e:
            print(f"[ERROR] Video processing failed: {e}")
            return {
                'url': video_url,
                'title': 'Error',
                'author': 'Error',
                'transcript': '',
                'summary': 'Processing failed',
                'processing_time': 0
            }

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
app = FastAPI(title="RTX 5090 Ultra-Optimized Video RSS")

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
    print("🚀 Starting RTX 5090 Ultra-Optimized Backend...")
    await transcription_engine.initialize_models()
    print("✅ RTX 5090 Backend ready!")

@app.get("/health")
async def health_check():
    """Health check with GPU status"""
    gpu_info = {
        "status": "healthy",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
        "gpu_memory_used": f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A",
        "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
    }
    return gpu_info

@app.get("/rss/bilibili", response_class=PlainTextResponse)
async def get_rtx5090_rss_feed(limit: int = 5):
    """RTX 5090 optimized RSS feed"""
    try:
        print(f"[RTX5090] Generating RSS feed with {limit} videos...")
        start_time = time.time()

        # Mock video URLs (replace with real Bilibili fetching)
        video_urls = [f"https://www.bilibili.com/video/BV{i:010d}" for i in range(1, limit + 1)]

        # Ultra-fast batch processing
        videos = await transcription_engine.batch_transcribe(video_urls)

        # Generate RSS
        rss = ET.Element("rss", version="2.0")
        rss.set("xmlns:content", "http://purl.org/rss/1.0/modules/content/")

        channel = ET.SubElement(rss, "channel")
        ET.SubElement(channel, "title").text = "Ultra-Fast AI Video Feed"
        ET.SubElement(channel, "description").text = "GPU-accelerated video transcription"
        ET.SubElement(channel, "link").text = "http://111.186.3.124:8000"
        ET.SubElement(channel, "lastBuildDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

        for video in videos:
            item = ET.SubElement(channel, "item")
            ET.SubElement(item, "title").text = video.get('title', 'No Title')
            ET.SubElement(item, "link").text = video.get('url', '')
            ET.SubElement(item, "description").text = f"🤖 AI总结：{video.get('summary', '')}\n📄 转录：{video.get('transcript', '')[:100]}..."
            ET.SubElement(item, "pubDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

        total_time = time.time() - start_time
        print(f"✅ RTX 5090 RSS generated in {total_time:.2f}s for {limit} videos")

        return ET.tostring(rss, encoding="unicode")

    except Exception as e:
        print(f"[ERROR] RSS generation failed: {e}")
        return f"<?xml version='1.0'?><rss version='2.0'><channel><title>Error</title><description>RTX 5090 Backend Error: {e}</description></channel></rss>"

if __name__ == "__main__":
    print("🚀 Starting RTX 5090 Ultra-Optimized Backend...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9001,  # Use port 9001 for RTX 5090 version
        log_level="info",
        access_log=False
    )