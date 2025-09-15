#!/usr/bin/env python3
"""
RTX 5090 Incremental RSS Backend with PCIe Optimization
Processes 1 video per minute and updates RSS feed incrementally
Features:
- PCIe optimizations with pinned memory
- Persistent GPU buffers
- Incremental RSS feed updates
- 1 video per minute processing rate
"""

import torch
import numpy as np
import asyncio
import aiohttp
from fastapi import FastAPI, Response
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import librosa
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import json
from pathlib import Path
import yt_dlp
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class IncrementalRTX5090Backend:
    """RTX 5090 backend with incremental RSS updates and PCIe optimization"""

    def __init__(self):
        self.device = "cuda:0"
        self.whisper_model = None
        self.llm_model = None
        self.llm_tokenizer = None

        # Incremental RSS feed storage
        self.processed_videos = []
        self.processing_lock = asyncio.Lock()
        self.background_task = None

        # Processing rate: 1 video per minute
        self.processing_interval = 60  # seconds

        logger.info("🚀 RTX 5090 Incremental Backend - Simplified GPU Processing")
        self.setup_gpu_optimization()

    def setup_gpu_optimization(self):
        """Configure GPU optimizations without pinned memory"""
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set memory fraction (95% for RTX 5090)
        torch.cuda.set_per_process_memory_fraction(0.95, device=0)

        # Enable non-blocking transfers
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"🎮 GPU: {gpu_name}")
            logger.info(f"📊 Memory: {gpu_memory:.1f}GB")
            logger.info(f"✅ Direct GPU Processing: Enabled")


    async def load_models(self):
        """Load models with PCIe-optimized configurations"""
        start_time = time.time()

        # Load Whisper
        logger.info("[PCIe-OPT] Loading Whisper...")
        self.whisper_model = whisper.load_model(
            "large-v3",
            device=self.device,
            download_root="/tmp/whisper_cache"
        )

        # Keep model on GPU
        self.whisper_model.eval()
        for param in self.whisper_model.parameters():
            param.requires_grad = False
            param.data = param.data.to(self.device)

        logger.info(f"✅ Whisper loaded in {time.time()-start_time:.2f}s")

        # Load Qwen with 4-bit quantization
        logger.info("[PCIe-OPT] Loading Qwen2.5-7B...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=False  # Keep on GPU
        )

        model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},  # Force to GPU 0
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        logger.info(f"✅ Qwen2.5-7B loaded in {time.time()-start_time:.2f}s")

        # Warm up
        await self.warmup_models()

        logger.info(f"🚀 Models ready in {time.time()-start_time:.2f}s")

    async def warmup_models(self):
        """Warm up models"""
        logger.info("[PCIe-OPT] Warming up...")

        # Warm up Whisper
        dummy_audio = torch.randn(16000, device=self.device)
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                _ = self.whisper_model.transcribe(
                    dummy_audio.cpu().numpy(),
                    language="zh",
                    fp16=True
                )

        # Warm up LLM
        dummy_text = "Test"
        inputs = self.llm_tokenizer(dummy_text, return_tensors="pt").to(self.device)
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                _ = self.llm_model.generate(**inputs, max_new_tokens=10)

        torch.cuda.empty_cache()
        logger.info("✅ Models warmed up")

    async def process_video_gpu(self, video: Dict) -> Dict:
        """Process a single video with GPU optimization"""
        try:
            logger.info(f"🎬 Processing: {video.get('title', 'Unknown')[:50]}...")
            start_time = time.time()

            # Download audio
            audio_data = await self.download_audio(video['url'])
            if not audio_data:
                video['transcript'] = "Failed to download audio"
                video['summary'] = "Processing failed"
                return video

            # Process audio using librosa to properly decode WAV format
            try:
                import librosa
                import io

                # Use librosa to load audio from bytes (handles WAV headers properly)
                audio_np, sample_rate = librosa.load(io.BytesIO(audio_data), sr=16000)

                # Ensure audio is in the right format for Whisper
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.mean(axis=0)  # Convert stereo to mono

                logger.info(f"Audio loaded: {len(audio_np)} samples at {sample_rate}Hz")

                # Transcribe directly
                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        result = self.whisper_model.transcribe(
                            audio_np,
                            language="zh",
                            fp16=True,
                            beam_size=1,
                            best_of=1,
                            temperature=0,
                            condition_on_previous_text=False
                        )
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                video['transcript'] = "Audio format error"
                video['summary'] = "Processing failed"
                return video

            transcript = result.get("text", "")
            video['transcript'] = transcript

            # Summarize with minimal transfers
            if transcript and len(transcript) > 10:
                prompt = f"请用一个详细段落总结以下内容，包含主要观点和关键信息：\n{transcript[:1000]}\n\n总结："

                inputs = self.llm_tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )

                # Keep on GPU
                input_ids = inputs["input_ids"].to(self.device, non_blocking=True)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, non_blocking=True)

                torch.cuda.synchronize()

                with torch.amp.autocast('cuda'):
                    with torch.no_grad():
                        outputs = self.llm_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=150,  # Increased for paragraph-style summaries
                            do_sample=False,
                            pad_token_id=self.llm_tokenizer.pad_token_id,
                            eos_token_id=self.llm_tokenizer.eos_token_id,
                            use_cache=True,
                            temperature=0.1  # Add small temperature for better generation
                        )

                        # Extract only the newly generated tokens (not the input prompt)
                        input_length = input_ids.shape[1]
                        new_tokens = outputs[:, input_length:]
                        outputs_cpu = new_tokens.cpu()

                # Decode only the generated part
                summary = self.llm_tokenizer.decode(outputs_cpu[0], skip_special_tokens=True).strip()

                # Clean up the summary
                if summary and len(summary) > 3:
                    video['summary'] = summary[:300]  # Allow longer paragraph summaries
                else:
                    # Fallback: create a simple summary from transcript
                    video['summary'] = f"视频内容：{transcript[:100]}..."
            else:
                video['summary'] = transcript[:200] if transcript else ""

            process_time = time.time() - start_time
            video['process_time'] = f"{process_time:.2f}s"
            video['processed_at'] = datetime.now().isoformat()

            logger.info(f"✅ Processed in {process_time:.2f}s")

            return video

        except Exception as e:
            logger.error(f"Processing error: {e}")
            video['transcript'] = "Error processing"
            video['summary'] = str(e)[:100]
            return video

    async def download_audio(self, url: str) -> bytes:
        """Download and extract audio from video"""
        import tempfile
        import os

        try:
            # Use a temporary file for audio extraction
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name

            ydl_opts = {
                'format': 'bestaudio/best',
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '16',
                }],
                'outtmpl': tmp_path.replace('.wav', '.%(ext)s'),
                'logtostderr': False,
            }

            # Actually download and process the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Read the processed audio file
            if os.path.exists(tmp_path):
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                os.unlink(tmp_path)  # Clean up
                return audio_data
            else:
                logger.error(f"Audio file not created: {tmp_path}")
                return None

        except Exception as e:
            logger.error(f"Audio download error: {e}")
            # Clean up any remaining temp files
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except:
                pass
            return None

    async def process_videos_incrementally(self):
        """Background task to process videos incrementally"""
        logger.info("🔄 Starting incremental processing (1 video/minute)...")

        while True:
            try:
                # Get real data
                from real_data_extractor import RealDataExtractor

                async with RealDataExtractor() as extractor:
                    real_data = await extractor.extract_all_real_data()
                    all_videos = real_data.get('bilibili', [])

                # Process videos not yet processed
                for video in all_videos:
                    video_id = video.get('id')

                    # Check if already processed
                    if any(v.get('id') == video_id for v in self.processed_videos):
                        continue

                    # Process video
                    logger.info(f"⏳ Processing video {len(self.processed_videos)+1}/{len(all_videos)}")
                    processed = await self.process_video_gpu(video)

                    # Add to processed list
                    async with self.processing_lock:
                        self.processed_videos.append(processed)
                        logger.info(f"📊 RSS feed updated: {len(self.processed_videos)} videos")

                    # Wait for next interval
                    logger.info(f"⏰ Waiting {self.processing_interval}s before next video...")
                    await asyncio.sleep(self.processing_interval)

                # If all videos processed, wait before checking for new ones
                logger.info("✅ All videos processed. Checking for new content in 5 minutes...")
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Incremental processing error: {e}")
                await asyncio.sleep(60)

    def generate_rss_feed(self, personalized: bool = True, full_transcription: bool = True) -> str:
        """Generate RSS feed from processed videos"""
        rss_items = []

        for video in self.processed_videos:
            # Include full transcript if requested
            transcript_content = video.get('transcript', '') if full_transcription else video.get('transcript', '')[:500] + '...'

            # Add personalization tag if enabled
            personalized_tag = "<p><strong>✨ Personalized Content</strong></p>" if personalized else ""

            rss_items.append(f"""
            <item>
                <title>{video.get('title', 'Unknown')}</title>
                <link>{video.get('url', '')}</link>
                <description>
                    <![CDATA[
                    {personalized_tag}
                    <h3>AI Summary</h3>
                    <p>{video.get('summary', 'No summary')}</p>
                    <h3>Transcript</h3>
                    <p>{transcript_content}</p>
                    <p><strong>Processing:</strong> RTX 5090 with PCIe Optimization</p>
                    <p><strong>Process Time:</strong> {video.get('process_time', 'N/A')}</p>
                    <p><strong>Processed At:</strong> {video.get('processed_at', '')}</p>
                    ]]>
                </description>
                <pubDate>{video.get('upload_date', datetime.now().isoformat())}</pubDate>
                <author>{video.get('author', 'Unknown')}</author>
            </item>
            """)

        rss_content = f"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>AI-Powered Video Feed</title>
                <description>Incremental feed with 1 video per minute updates</description>
                <link>http://111.186.3.124:9001</link>
                <lastBuildDate>{datetime.now().isoformat()}</lastBuildDate>
                <ttl>1</ttl>
                <generator>RTX 5090 Incremental Backend</generator>
                {"".join(rss_items)}
            </channel>
        </rss>
        """

        return rss_content

# Global instance
backend = IncrementalRTX5090Backend()

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    await backend.load_models()
    # Start background processing
    backend.background_task = asyncio.create_task(backend.process_videos_incrementally())
    logger.info("✅ Incremental backend ready!")

@app.get("/rss/bilibili")
async def get_bilibili_rss(
    limit: int = 10,
    personalized: bool = True,  # Default to True
    full_transcription: bool = True  # Default to True
):
    """Get incrementally updated RSS feed"""
    logger.info(f"📡 RSS request: {len(backend.processed_videos)} videos available")
    logger.info(f"   Personalized: {personalized}, Full Transcription: {full_transcription}")

    rss_content = backend.generate_rss_feed(personalized, full_transcription)
    return Response(content=rss_content, media_type="application/rss+xml")

@app.get("/stats")
async def get_stats():
    """Get processing stats"""
    stats = {
        "processed_videos": len(backend.processed_videos),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
        "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
        "processing_rate": "1 video per minute",
        "optimizations": [
            "Direct GPU Processing",
            "Non-blocking Transfers",
            "Simplified Memory Management",
            "Buffer Alignment Fixed"
        ]
    }
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)