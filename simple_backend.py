#!/usr/bin/env python3
"""
Simple Backend Server for Video RSS Aggregator Demo
Provides mock data for frontend demonstration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import json
import os
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file if it exists
def load_env_file():
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key.strip() not in os.environ:
                        os.environ[key.strip()] = value.strip()
        print(f"[OK] Loaded credentials from .env file")
        if os.getenv('BILIBILI_SESSDATA') and os.getenv('BILIBILI_SESSDATA') not in ['your_sessdata_here', 'demo_mode']:
            print("   - Bilibili authentication configured")

# Load credentials at startup
load_env_file()

app = FastAPI(title="Video RSS Aggregator API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Real data from Chinese video platforms
import aiohttp
import asyncio
from typing import Optional
from rust_video_core import BilibiliClient, RssGenerator, ContentAnalyzer

# Real video data cache
REAL_VIDEO_CACHE = {
    "bilibili": [],
    "douyin": [],
    "kuaishou": []
}

@app.get("/")
async def root():
    return {"message": "Video RSS Aggregator API", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    # Check if Bilibili credentials are configured
    has_bilibili_auth = False
    sessdata = os.getenv('BILIBILI_SESSDATA', '')
    if sessdata and sessdata not in ['', 'your_sessdata_here', 'demo_mode']:
        has_bilibili_auth = True

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bilibili_auth": has_bilibili_auth,
        "auth_mode": "authenticated" if has_bilibili_auth else "public_api_only"
    }

@app.get("/api/health")
async def api_health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0"}

@app.get("/api/cache/stats")
async def get_cache_stats():
    return {
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_hit_rate": 0.0,
        "total_requests": 0,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/manifest.json")
async def get_manifest():
    return {
        "short_name": "AI Video RSS",
        "name": "AI Video RSS Aggregator",
        "icons": [
            {
                "src": "favicon.ico",
                "sizes": "64x64 32x32 24x24 16x16",
                "type": "image/x-icon"
            }
        ],
        "start_url": ".",
        "display": "standalone",
        "theme_color": "#000000",
        "background_color": "#ffffff"
    }

async def refresh_real_data():
    """Refresh real data cache using Rust-powered extractor"""
    try:
        print("[INFO] Refreshing real data cache with Rust-powered extraction...")

        # Use Rust-powered Bilibili client for high-performance data extraction
        credentials = {}
        sessdata = os.getenv('BILIBILI_SESSDATA', '')
        bili_jct = os.getenv('BILIBILI_BILI_JCT', '')
        buvid3 = os.getenv('BILIBILI_BUVID3', '')

        if sessdata and sessdata not in ['', 'your_sessdata_here', 'demo_mode']:
            credentials = {
                'sessdata': sessdata,
                'bili_jct': bili_jct,
                'buvid3': buvid3
            }

        from rust_video_core import FetchOptions
        client = BilibiliClient(credentials if credentials else None)

        # Fetch recommendations using Rust implementation
        options = FetchOptions(
            limit=20,
            include_transcription=False,
            personalized=bool(credentials),
            credentials=credentials if credentials else None
        )
        bilibili_videos = await client.fetch_recommendations(options)
        REAL_VIDEO_CACHE["bilibili"] = bilibili_videos

        # For other platforms, try fallback extraction
        try:
            from real_data_extractor import RealDataExtractor
            async with RealDataExtractor() as extractor:
                douyin_data = await extractor.extract_douyin_data()
                kuaishou_data = await extractor.extract_kuaishou_data()
                REAL_VIDEO_CACHE["douyin"] = douyin_data
                REAL_VIDEO_CACHE["kuaishou"] = kuaishou_data
        except:
            print("[INFO] Fallback extractors not available, using Bilibili data only")

        total_videos = sum(len(v) for v in REAL_VIDEO_CACHE.values())
        print(f"[OK] Real data cache refreshed: {total_videos} total videos (Rust-powered)")

    except Exception as e:
        print(f"[ERROR] Rust data extraction failed: {e}")
        await populate_sample_data()

async def populate_sample_data():
    """No sample data - real data only"""
    # Remove all generated/mock data as requested
    REAL_VIDEO_CACHE["bilibili"] = []
    REAL_VIDEO_CACHE["douyin"] = []
    REAL_VIDEO_CACHE["kuaishou"] = []
    print("[WARNING] No sample data - real data collection required")

@app.get("/api/videos/{platform}")
async def get_platform_videos(platform: str, limit: int = 10, include_summary: bool = False):
    """Get real videos from specified platform"""
    if platform not in ["bilibili", "douyin", "kuaishou", "youtube", "all"]:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")

    # Refresh cache if empty
    if not any(REAL_VIDEO_CACHE.values()):
        await refresh_real_data()

    # Get videos from cache
    if platform == "all":
        all_videos = []
        for videos in REAL_VIDEO_CACHE.values():
            all_videos.extend(videos)
        videos = all_videos[:limit]
    elif platform == "youtube":
        from real_data_collector import RealDataCollector
        async with RealDataCollector() as collector:
            videos = await collector.get_youtube_rss_videos("UCBa659QWEk1AI4Tg--mrJ2A", limit)
    else:
        videos = REAL_VIDEO_CACHE.get(platform, [])[:limit]

    return {
        "platform": platform,
        "count": len(videos),
        "videos": videos,
        "include_summary": include_summary,
        "timestamp": datetime.now().isoformat(),
        "data_source": "real_data"
    }

@app.get("/api/recommendations/bilibili")
async def get_bilibili_recommendations(limit: int = 20, include_transcription: bool = False):
    """Get personalized Bilibili recommendations using Rust-powered client"""
    try:
        print(f"[INFO] Fetching {limit} personalized Bilibili recommendations with Rust client...")

        # Get credentials from environment
        credentials = {}
        sessdata = os.getenv('BILIBILI_SESSDATA', '')
        bili_jct = os.getenv('BILIBILI_BILI_JCT', '')
        buvid3 = os.getenv('BILIBILI_BUVID3', '')

        if sessdata and sessdata not in ['', 'your_sessdata_here', 'demo_mode']:
            credentials = {
                'sessdata': sessdata,
                'bili_jct': bili_jct,
                'buvid3': buvid3
            }

        # Use Rust-powered BilibiliClient for high performance
        from rust_video_core import FetchOptions
        client = BilibiliClient(credentials if credentials else None)

        options = FetchOptions(
            limit=limit,
            include_transcription=include_transcription,
            personalized=bool(credentials),
            credentials=credentials if credentials else None
        )

        recommendations = await client.fetch_recommendations(options)

        print(f"[SUCCESS] Retrieved {len(recommendations)} personalized recommendations (Rust-powered)")

        return {
            "platform": "bilibili",
            "type": "personalized_recommendations",
            "count": len(recommendations),
            "videos": recommendations,
            "timestamp": datetime.now().isoformat(),
            "data_source": "rust_bilibili_client",
            "authenticated": bool(credentials),
            "transcription_enabled": include_transcription
        }

    except Exception as e:
        print(f"[ERROR] Failed to fetch personalized recommendations: {e}")
        return {
            "platform": "bilibili",
            "type": "personalized_recommendations",
            "count": 0,
            "videos": [],
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "authenticated": False,
            "transcription_enabled": False
        }

@app.get("/api/platforms")
async def get_supported_platforms():
    """Get list of supported platforms"""
    return {
        "platforms": [
            {"id": "youtube", "name": "YouTube", "mau": "2.7B", "focus": "Global, All Content"},
            {"id": "bilibili", "name": "哔哩哔哩 (Bilibili)", "mau": "340M", "focus": "Gen Z, 学习娱乐"},
            {"id": "douyin", "name": "抖音 (Douyin)", "mau": "766M", "focus": "短视频, 全年龄段"},
            {"id": "kuaishou", "name": "快手 (Kuaishou)", "dau": "408M", "focus": "下沉市场, 生活记录"}
        ],
        "total": 4
    }

@app.get("/api/analyze/video")
async def analyze_single_video(video_url: str):
    """Analyze single video using Rust-powered content analysis"""
    if not video_url:
        raise HTTPException(status_code=400, detail="video_url parameter required")

    try:
        # Use Rust-powered BilibiliClient to fetch video info
        client = BilibiliClient()
        video_info = await client.fetch_video_info(video_url)

        # Use Rust-powered ContentAnalyzer for high-performance analysis
        analyzer = ContentAnalyzer()
        analysis_result = await analyzer.analyze_video(video_info)

        return {
            "video_info": video_info,
            "analysis_result": analysis_result,
            "legal_compliance": "Fair Use Academic Research",
            "analysis_method": "Rust-powered Content Analysis",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/analyze/batch")
async def analyze_batch_videos(video_urls: List[str]):
    """Batch analyze multiple videos using Rust-powered analysis"""
    if not video_urls:
        raise HTTPException(status_code=400, detail="video_urls required")

    if len(video_urls) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 videos per batch")

    try:
        # Use Rust-powered clients for high-performance batch processing
        client = BilibiliClient()
        analyzer = ContentAnalyzer()

        # Fetch video info for all URLs
        video_infos = []
        for url in video_urls:
            try:
                video_info = await client.fetch_video_info(url)
                video_infos.append(video_info)
            except Exception as e:
                print(f"[WARNING] Failed to fetch video info for {url}: {e}")

        # Batch analyze all videos using Rust
        results = await analyzer.batch_analyze(video_infos)

        return {
            "batch_results": results,
            "total_analyzed": len(results),
            "legal_compliance": "Fair Use Academic Research",
            "analysis_method": "Rust-powered Batch Analysis",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "error": f"Batch analysis failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/research/trending/{platform}")
async def get_research_trending(platform: str, limit: int = 10):
    """Get trending content for academic research - Legal API access"""
    from chinese_api_integration import ChineseAPIIntegration

    if platform not in ["bilibili", "douyin", "kuaishou"]:
        raise HTTPException(status_code=400, detail=f"Unsupported platform for research: {platform}")

    try:
        async with ChineseAPIIntegration() as api:
            if platform == "bilibili":
                data = await api.get_bilibili_trending(limit)
            elif platform == "douyin":
                data = await api.get_douyin_research_data(limit)
            elif platform == "kuaishou":
                data = await api.get_kuaishou_research_data(limit)

            return {
                "platform": platform,
                "research_data": data,
                "count": len(data),
                "legal_framework": "Fair Use Academic Research",
                "data_source": "Official APIs + Legal Integration",
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        return {
            "error": f"Research data collection failed: {str(e)}",
            "platform": platform,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/research/comprehensive")
async def get_comprehensive_research():
    """Get comprehensive research data from all Chinese platforms"""
    from chinese_api_integration import ChineseAPIIntegration

    try:
        async with ChineseAPIIntegration() as api:
            research_data = await api.get_comprehensive_research_data()

            return {
                "comprehensive_research": research_data,
                "legal_compliance": "Fair Use Academic Research Framework",
                "methodology": "Official APIs + Content Analysis + Streaming Metadata",
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        return {
            "error": f"Comprehensive research failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/stream/{platform}")
async def stream_platform_videos(platform: str, limit: int = 10):
    """Stream videos from platform in real-time - REAL DATA ONLY"""
    if platform not in ["youtube", "bilibili", "douyin", "kuaishou"]:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")

    # No generated data - only real streaming data
    return {
        "platform": platform,
        "method": "streaming",
        "count": 0,
        "videos": [],
        "message": "Real streaming data only - requires active internet connection and valid URLs",
        "legal_note": "Use /api/research/trending/{platform} for legal API-based data collection",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/rss/{platform}")
async def get_rss_feed(platform: str, limit: int = 10, summary: bool = True, personalized: bool = False, full_transcription: bool = False):
    """Generate RSS feed for platform - REAL DATA ONLY"""
    if platform not in ["bilibili", "douyin", "kuaishou", "all"]:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")

    # Use personalized recommendations if requested and available
    if personalized and platform == "bilibili":
        try:
            print(f"[INFO] Using Rust-powered personalized Bilibili recommendations for RSS feed...")

            # Get credentials from environment
            credentials = {}
            sessdata = os.getenv('BILIBILI_SESSDATA', '')
            bili_jct = os.getenv('BILIBILI_BILI_JCT', '')
            buvid3 = os.getenv('BILIBILI_BUVID3', '')

            if sessdata and sessdata not in ['', 'your_sessdata_here', 'demo_mode']:
                credentials = {
                    'sessdata': sessdata,
                    'bili_jct': bili_jct,
                    'buvid3': buvid3
                }

            from rust_video_core import FetchOptions
            client = BilibiliClient(credentials if credentials else None)

            options = FetchOptions(
                limit=limit,
                include_transcription=full_transcription,
                personalized=bool(credentials),
                credentials=credentials if credentials else None
            )
            videos = await client.fetch_recommendations(options)

            print(f"[SUCCESS] Retrieved {len(videos)} personalized videos for RSS (Rust-powered)")
        except Exception as e:
            print(f"[ERROR] Personalized fetch failed: {e}")
            # Fallback to cache
            if not any(REAL_VIDEO_CACHE.values()):
                await refresh_real_data()
            videos = REAL_VIDEO_CACHE.get(platform, [])[:limit]
    else:
        # Get videos from real data cache (no mock data)
        if not any(REAL_VIDEO_CACHE.values()):
            await refresh_real_data()

        if platform == "all":
            videos = []
            for platform_videos in REAL_VIDEO_CACHE.values():
                videos.extend(platform_videos)
            videos = videos[:limit]
        else:
            videos = REAL_VIDEO_CACHE.get(platform, [])[:limit]

    # Use Rust-powered RSS generation for high performance
    try:
        print(f"[INFO] Generating RSS feed using Rust-powered RSS generator...")

        # Configure RSS generator
        rss_config = {
            "title": f"AI智能内容摘要 - {platform.title()} 精选视频",
            "description": "基于人工智能技术的中文视频平台内容聚合与智能分析 - Rust性能加速",
            "link": "http://localhost:3000",
            "language": "zh-CN",
            "generator": "Rust Video RSS Core v1.0"
        }

        generator = RssGenerator(rss_config)

        # Generate RSS feed using Rust implementation
        if summary:
            rss_xml = generator.generate_feed_with_summary(videos, include_ai_summary=True)
        else:
            rss_xml = generator.generate_feed(videos)

        print(f"[SUCCESS] Generated RSS feed with {len(videos)} videos (Rust-powered)")
        return Response(content=rss_xml, media_type="application/xml")

    except Exception as e:
        print(f"[ERROR] Rust RSS generation failed: {e}")

        # Fallback to manual XML generation
        rss_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
    <channel>
        <title>AI智能内容摘要 - {platform.title()} 精选视频</title>
        <description>基于人工智能技术的中文视频平台内容聚合与智能分析 - 备用模式</description>
        <link>http://localhost:3000</link>
        <lastBuildDate>{datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')}</lastBuildDate>
        <generator>Video RSS Aggregator Fallback v1.0</generator>
    </channel>
</rss>'''

        return Response(content=rss_xml, media_type="application/xml")

# Import Response for RSS endpoint
from fastapi.responses import Response
import random
import re
import hashlib

def _generate_transcription_content(video: Dict[str, Any]) -> str:
    """Generate HTML content for full transcription data in RSS"""
    transcription = video.get('full_transcription', {})

    if not transcription:
        return ""

    content_parts = []

    # Add paragraph summary
    paragraph_summary = transcription.get('paragraph_summary', '')
    if paragraph_summary:
        content_parts.append(f'<p><strong>📄 完整段落摘要：</strong>{paragraph_summary}</p>')

    # Add sentence subtitle
    sentence_subtitle = transcription.get('sentence_subtitle', '')
    if sentence_subtitle:
        content_parts.append(f'<p><strong>📝 字幕句子：</strong>{sentence_subtitle}</p>')

    # Add truncated transcript preview
    full_transcript = transcription.get('full_transcript', '')
    if full_transcript:
        # Show first 200 characters of transcript
        transcript_preview = full_transcript[:200] + "..." if len(full_transcript) > 200 else full_transcript
        content_parts.append(f'<p><strong>📰 转录内容：</strong>{transcript_preview}</p>')

    # Add model information
    model_info = transcription.get('model_info', {})
    if model_info:
        transcriber = model_info.get('transcriber', 'unknown')
        summarizer = model_info.get('summarizer', 'unknown')
        status = transcription.get('status', 'unknown')
        content_parts.append(f'<p><strong>🤖 AI模型信息：</strong>状态: {status} | 转录器: {transcriber} | 摘要器: {summarizer}</p>')

    return "\n                ".join(content_parts)

async def generate_content_summary(video: Dict[str, Any]) -> str:
    """Generate content-rich summary using Rust-powered ContentAnalyzer"""
    try:
        analyzer = ContentAnalyzer()

        # Use Rust-powered content analysis for high performance
        summary_result = await analyzer.analyze_video(video)

        if summary_result and 'ai_summary' in summary_result:
            return summary_result['ai_summary']
        elif summary_result and 'keywords' in summary_result:
            # Build summary from keywords and sentiment
            keywords = summary_result.get('keywords', [])[:5]
            content_type = summary_result.get('content_type', 'Entertainment')

            summary_parts = []
            summary_parts.append(f"📦 内容类型：{content_type}")

            if keywords:
                summary_parts.append(f"🏷️ 关键词：{', '.join(keywords)}")

            return ' | '.join(summary_parts)

        # Fallback to metadata-based summary
        return generate_metadata_summary(video)

    except Exception as e:
        print(f"Rust content analysis error: {e}")
        return generate_metadata_summary(video)

def generate_metadata_summary(video: Dict[str, Any]) -> str:
    """Generate informative summary from video metadata when content analysis fails"""
    title = video.get('title', '').strip()
    description = video.get('description', '').strip()
    tags = video.get('tags', [])
    duration = video.get('duration', '')
    view_count = video.get('view_count', 0)
    author = video.get('author', '').strip()

    summary_parts = []

    # Extract key information from title
    if title:
        # Identify content type from title
        if any(keyword in title for keyword in ['合集', '集合', '精选', '推荐']):
            summary_parts.append(f"📦 内容合集：{title}")
        elif any(keyword in title for keyword in ['教程', '攻略', '方法', '技巧']):
            summary_parts.append(f"📚 实用教程：{title}")
        elif any(keyword in title for keyword in ['音乐', '歌曲', '歌单']):
            summary_parts.append(f"🎵 音乐内容：{title}")
        elif any(keyword in title for keyword in ['游戏', '实况', '解说']):
            summary_parts.append(f"🎮 游戏内容：{title}")
        else:
            summary_parts.append(f"📹 视频内容：{title}")

    # Add duration and engagement info
    if duration and view_count:
        summary_parts.append(f"⏱️ 时长{duration}，已有{view_count:,}人观看")
    elif duration:
        summary_parts.append(f"⏱️ 视频时长{duration}")
    elif view_count:
        summary_parts.append(f"[STATS] 已有{view_count:,}人观看")

    # Add description summary if available
    if description and description != '-' and len(description) > 10:
        # Take first meaningful part of description
        desc_summary = description[:80] + ('...' if len(description) > 80 else '')
        summary_parts.append(f"📝 简介：{desc_summary}")

    # Add tags/topics
    if tags and len(tags) > 0:
        main_tags = tags[:3]  # First 3 tags
        summary_parts.append(f"🏷️ 相关话题：{', '.join(main_tags)}")

    # Add creator info
    if author:
        summary_parts.append(f"👤 创作者：{author}")

    return ' | '.join(summary_parts) if summary_parts else f"📹 {title or '视频内容'}"

if __name__ == "__main__":
    import uvicorn
    print("[START] Starting Video RSS Aggregator API Server")
    print("[FRONTEND] Frontend: http://localhost:3000")
    print("[BACKEND] Backend: http://localhost:8000")
    print("[DOCS] API Docs: http://localhost:8000/docs")
    print("[READY] AI Content Digest Platform Ready!")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)