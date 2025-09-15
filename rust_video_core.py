#!/usr/bin/env python3
"""
Python wrapper for Rust video RSS core functionality.
Falls back to Python implementation when Rust library is not available.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
import xml.etree.ElementTree as ET

# Try to import Rust library, fallback to Python implementation
try:
    import video_rss_core
    HAS_RUST = True
    print("[INFO] Using Rust-powered video RSS core")
except ImportError:
    HAS_RUST = False
    print("[INFO] Using Python fallback for video RSS core")

class Platform(Enum):
    BILIBILI = "bilibili"
    DOUYIN = "douyin"
    KUAISHOU = "kuaishou"
    YOUTUBE = "youtube"

class TranscriptionStatus(Enum):
    SUCCESS = "success"
    PENDING = "pending"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"

class ContentType(Enum):
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    GAMING = "gaming"
    TECHNOLOGY = "technology"
    MUSIC = "music"
    OTHER = "other"

@dataclass
class ModelInfo:
    transcriber: str
    summarizer: str

@dataclass
class TranscriptionData:
    paragraph_summary: str
    sentence_subtitle: str
    full_transcript: str
    status: TranscriptionStatus
    model_info: ModelInfo
    source_types: List[str]

@dataclass
class VideoInfo:
    id: str
    title: str
    description: str
    url: str
    author: str
    upload_date: datetime
    duration: Optional[int] = None
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    tags: List[str] = None
    thumbnail_url: Optional[str] = None
    platform: Platform = Platform.BILIBILI
    transcription: Optional[TranscriptionData] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class ContentSummary:
    ai_summary: str
    keywords: List[str]
    sentiment: Optional[float] = None
    content_type: ContentType = ContentType.OTHER

@dataclass
class FetchOptions:
    limit: int = 10
    include_transcription: bool = False
    personalized: bool = False
    credentials: Optional[Dict[str, str]] = None

@dataclass
class RssConfig:
    title: str = "AI智能内容摘要 - 精选视频"
    description: str = "基于人工智能技术的中文视频平台内容聚合与智能分析"
    link: str = "http://localhost:3000"
    language: str = "zh-CN"
    generator: str = "Video RSS Core v1.0"

class BilibiliClient:
    """High-performance Bilibili client with Rust backend when available"""

    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        self.credentials = credentials

        if HAS_RUST:
            self._rust_client = video_rss_core.PyBilibiliClient(credentials)
        else:
            import aiohttp
            self._setup_python_client()

    def _setup_python_client(self):
        """Setup Python HTTP client as fallback"""
        import aiohttp

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/html',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.bilibili.com'
        }

        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30)

        # Add cookies if credentials provided
        if self.credentials:
            cookie_jar = aiohttp.CookieJar()
            self._client = aiohttp.ClientSession(
                headers=headers,
                connector=connector,
                timeout=timeout,
                cookie_jar=cookie_jar
            )

            # Set cookies
            self._client.cookie_jar.update_cookies({
                'SESSDATA': self.credentials.get('sessdata', ''),
                'bili_jct': self.credentials.get('bili_jct', ''),
                'buvid3': self.credentials.get('buvid3', '')
            })
        else:
            self._client = aiohttp.ClientSession(
                headers=headers,
                connector=connector,
                timeout=timeout
            )

    async def fetch_recommendations(self, options: FetchOptions) -> List[VideoInfo]:
        """Fetch video recommendations with high performance"""
        if HAS_RUST:
            # Use Rust implementation for maximum performance
            py_videos = await self._rust_client.fetch_recommendations(
                options.limit,
                options.include_transcription,
                options.personalized
            )
            return [self._dict_to_video_info(video) for video in py_videos]
        else:
            # Python fallback implementation
            return await self._fetch_recommendations_python(options)

    async def _fetch_recommendations_python(self, options: FetchOptions) -> List[VideoInfo]:
        """Python implementation for video fetching"""
        api_url = ("https://api.bilibili.com/x/web-interface/index/top/rcmd"
                  if options.personalized
                  else "https://api.bilibili.com/x/web-interface/ranking/v2")

        params = {"rid": "0", "day": "3", "arc_type": "0"}

        async with self._client.get(api_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"API request failed: {response.status}")

            data = await response.json()

            if data.get('code') != 0:
                raise Exception(f"API error: {data.get('message', 'Unknown error')}")

            videos_data = data['data']['item' if options.personalized else 'list']

            videos = []
            for video_data in videos_data[:options.limit]:
                try:
                    video = self._parse_video_info(video_data)
                    videos.append(video)
                except Exception as e:
                    print(f"[WARNING] Failed to parse video: {e}")
                    continue

            return videos

    def _parse_video_info(self, video_data: Dict[str, Any]) -> VideoInfo:
        """Parse video info from API response"""
        video_id = video_data.get('bvid') or f"av{video_data.get('aid', '')}"
        title = video_data.get('title', 'Untitled')
        description = video_data.get('desc', '')
        url = f"https://www.bilibili.com/video/{video_id}"
        author = video_data.get('owner', {}).get('name', 'Unknown')

        upload_timestamp = video_data.get('pubdate', 0)
        upload_date = datetime.fromtimestamp(upload_timestamp, tz=timezone.utc) if upload_timestamp else datetime.now(timezone.utc)

        duration = video_data.get('duration')
        if isinstance(duration, str) and ':' in duration:
            duration = self._parse_duration(duration)

        stats = video_data.get('stat', {})
        view_count = stats.get('view', 0)
        like_count = stats.get('like', 0)
        comment_count = stats.get('reply', 0)

        tags = [tag.get('tag_name', '') for tag in video_data.get('tag', [])]

        thumbnail_url = video_data.get('pic', '')
        if thumbnail_url and thumbnail_url.startswith('//'):
            thumbnail_url = f"https:{thumbnail_url}"

        return VideoInfo(
            id=video_id,
            title=title,
            description=description,
            url=url,
            author=author,
            upload_date=upload_date,
            duration=duration,
            view_count=view_count,
            like_count=like_count,
            comment_count=comment_count,
            tags=tags,
            thumbnail_url=thumbnail_url,
            platform=Platform.BILIBILI
        )

    def _parse_duration(self, duration_str: str) -> Optional[int]:
        """Parse duration string to seconds"""
        try:
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 2:
                    return int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return int(duration_str)
        except (ValueError, IndexError):
            return None

    def _dict_to_video_info(self, data: Dict[str, Any]) -> VideoInfo:
        """Convert dictionary to VideoInfo object"""
        upload_date = datetime.fromisoformat(data['upload_date'].replace('Z', '+00:00'))

        transcription = None
        if 'transcription' in data and data['transcription']:
            trans_data = data['transcription']
            transcription = TranscriptionData(
                paragraph_summary=trans_data.get('paragraph_summary', ''),
                sentence_subtitle=trans_data.get('sentence_subtitle', ''),
                full_transcript=trans_data.get('full_transcript', ''),
                status=TranscriptionStatus(trans_data.get('status', 'unavailable')),
                model_info=ModelInfo(
                    transcriber=trans_data.get('model_info', {}).get('transcriber', ''),
                    summarizer=trans_data.get('model_info', {}).get('summarizer', '')
                ),
                source_types=trans_data.get('source_types', [])
            )

        return VideoInfo(
            id=data['id'],
            title=data['title'],
            description=data['description'],
            url=data['url'],
            author=data['author'],
            upload_date=upload_date,
            duration=data.get('duration'),
            view_count=data.get('view_count', 0),
            like_count=data.get('like_count', 0),
            comment_count=data.get('comment_count', 0),
            tags=data.get('tags', []),
            thumbnail_url=data.get('thumbnail_url'),
            platform=Platform(data.get('platform', 'bilibili')),
            transcription=transcription
        )

    async def close(self):
        """Close the client"""
        if not HAS_RUST and hasattr(self, '_client'):
            await self._client.close()

class RssGenerator:
    """High-performance RSS generator with Rust backend when available"""

    def __init__(self, config: RssConfig = None):
        self.config = config or RssConfig()

        if HAS_RUST:
            config_dict = asdict(self.config)
            self._rust_generator = video_rss_core.PyRssGenerator(config_dict)

    def generate_feed(self, videos: List[VideoInfo]) -> str:
        """Generate RSS feed from video list"""
        if HAS_RUST:
            # Use Rust implementation for maximum performance
            video_dicts = [self._video_info_to_dict(video) for video in videos]
            return self._rust_generator.generate_feed(video_dicts)
        else:
            # Python fallback implementation
            return self._generate_feed_python(videos)

    def generate_feed_with_summary(self, videos: List[VideoInfo], include_ai_summary: bool = True) -> str:
        """Generate RSS feed with AI summaries"""
        if HAS_RUST:
            video_dicts = [self._video_info_to_dict(video) for video in videos]
            return self._rust_generator.generate_feed_with_summary(video_dicts, include_ai_summary)
        else:
            return self._generate_feed_python(videos, include_ai_summary)

    def _generate_feed_python(self, videos: List[VideoInfo], include_ai_summary: bool = False) -> str:
        """Python implementation for RSS generation"""
        root = ET.Element('rss', {'version': '2.0', 'xmlns:content': 'http://purl.org/rss/1.0/modules/content/'})
        channel = ET.SubElement(root, 'channel')

        # Channel metadata
        ET.SubElement(channel, 'title').text = self.config.title
        ET.SubElement(channel, 'description').text = self.config.description
        ET.SubElement(channel, 'link').text = self.config.link
        ET.SubElement(channel, 'language').text = self.config.language
        ET.SubElement(channel, 'generator').text = self.config.generator
        ET.SubElement(channel, 'lastBuildDate').text = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S %z')

        # Add items
        for video in videos:
            item = ET.SubElement(channel, 'item')

            ET.SubElement(item, 'title').text = self._escape_xml(video.title)
            ET.SubElement(item, 'link').text = video.url
            ET.SubElement(item, 'guid', {'isPermaLink': 'true'}).text = video.url
            ET.SubElement(item, 'author').text = self._escape_xml(video.author)
            ET.SubElement(item, 'pubDate').text = video.upload_date.strftime('%a, %d %b %Y %H:%M:%S %z')

            # Generate description
            description = self._generate_item_description(video, include_ai_summary)
            ET.SubElement(item, 'description').text = self._escape_xml(description)

            # Add categories from tags
            for tag in video.tags[:5]:  # Limit to 5 tags
                ET.SubElement(item, 'category').text = self._escape_xml(tag)

        # Convert to string
        return ET.tostring(root, encoding='unicode', xml_declaration=True)

    def _generate_item_description(self, video: VideoInfo, include_ai_summary: bool = False) -> str:
        """Generate description for RSS item"""
        parts = []

        # Basic info
        parts.append(f"👤 作者：{video.author}")
        parts.append(f"👁️ 观看：{video.view_count:,} | 👍 点赞：{video.like_count:,}")

        if video.duration:
            minutes = video.duration // 60
            seconds = video.duration % 60
            parts.append(f"⏱️ 时长：{minutes:02d}:{seconds:02d}")

        if video.tags:
            tags_str = ', '.join(video.tags[:5])
            parts.append(f"🏷️ 标签：{tags_str}")

        if video.description:
            desc = video.description[:500] + ('...' if len(video.description) > 500 else '')
            parts.append(f"📝 简介：{desc}")

        # AI summary
        if include_ai_summary:
            ai_summary = self._generate_ai_summary(video)
            if ai_summary:
                parts.append(f"🤖 AI摘要：{ai_summary}")

        # Transcription data
        if video.transcription:
            if video.transcription.paragraph_summary:
                parts.append(f"📄 段落摘要：{video.transcription.paragraph_summary}")
            if video.transcription.sentence_subtitle:
                parts.append(f"📝 字幕：{video.transcription.sentence_subtitle}")

        parts.append(f"[SOURCE] 数据来源：{video.platform.value.upper()} - Real Data Only")

        return ' | '.join(parts)

    def _generate_ai_summary(self, video: VideoInfo) -> str:
        """Generate AI summary for video"""
        title_lower = video.title.lower()

        if '教程' in title_lower or '攻略' in title_lower:
            return f"📚 实用教程：{video.title[:50]}"
        elif '音乐' in title_lower or '歌曲' in title_lower:
            return f"🎵 音乐内容：{video.title[:50]}"
        elif '游戏' in title_lower or '实况' in title_lower:
            return f"🎮 游戏内容：{video.title[:50]}"
        else:
            return f"📹 视频内容：{video.title[:50]}"

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&apos;'))

    def _video_info_to_dict(self, video: VideoInfo) -> Dict[str, Any]:
        """Convert VideoInfo to dictionary for Rust interface"""
        data = asdict(video)
        data['upload_date'] = video.upload_date.isoformat()
        data['platform'] = video.platform.value

        if video.transcription:
            trans_dict = asdict(video.transcription)
            trans_dict['status'] = video.transcription.status.value
            data['transcription'] = trans_dict

        return data

class ContentAnalyzer:
    """High-performance content analyzer with Rust backend when available"""

    def __init__(self):
        if HAS_RUST:
            self._rust_analyzer = video_rss_core.PyContentAnalyzer()

    async def analyze_video(self, video: VideoInfo) -> ContentSummary:
        """Analyze video content"""
        if HAS_RUST:
            video_dict = self._video_info_to_dict(video)
            summary_dict = await self._rust_analyzer.analyze_video(video_dict)
            return self._dict_to_content_summary(summary_dict)
        else:
            return self._analyze_video_python(video)

    async def batch_analyze(self, videos: List[VideoInfo]) -> List[Dict[str, Any]]:
        """Batch analyze videos"""
        if HAS_RUST:
            video_dicts = [self._video_info_to_dict(video) for video in videos]
            return await self._rust_analyzer.batch_analyze(video_dicts)
        else:
            results = []
            for video in videos:
                start_time = time.time()
                summary = self._analyze_video_python(video)
                processing_time = int((time.time() - start_time) * 1000)

                results.append({
                    'video': asdict(video),
                    'summary': asdict(summary),
                    'processing_time_ms': processing_time
                })

                # Rate limiting
                await asyncio.sleep(0.1)

            return results

    def _analyze_video_python(self, video: VideoInfo) -> ContentSummary:
        """Python implementation for content analysis"""
        keywords = self._extract_keywords(video.title, video.description, video.tags)
        content_type = self._classify_content(video)
        sentiment = self._analyze_sentiment(video.title, video.description)
        ai_summary = self._generate_content_summary(video)

        return ContentSummary(
            ai_summary=ai_summary,
            keywords=keywords,
            sentiment=sentiment,
            content_type=content_type
        )

    def _extract_keywords(self, title: str, description: str, tags: List[str]) -> List[str]:
        """Extract keywords from content"""
        keywords = list(tags[:5])  # Start with tags

        # Extract Chinese words from title
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]{2,6}')
        title_words = chinese_pattern.findall(title)
        keywords.extend(title_words[:3])

        # Extract from description
        if description:
            desc_words = chinese_pattern.findall(description)
            keywords.extend(desc_words[:3])

        # Remove duplicates and common words
        stop_words = {'这个', '那个', '可以', '我们', '他们', '什么', '怎么', '非常', '真的', '还是'}
        keywords = [kw for kw in keywords if kw not in stop_words]

        return list(dict.fromkeys(keywords))[:10]  # Remove duplicates, limit to 10

    def _classify_content(self, video: VideoInfo) -> ContentType:
        """Classify video content type"""
        text = f"{video.title} {video.description} {' '.join(video.tags)}".lower()

        if any(word in text for word in ['教程', '学习', '教育']):
            return ContentType.EDUCATIONAL
        elif any(word in text for word in ['游戏', '电竞', '实况']):
            return ContentType.GAMING
        elif any(word in text for word in ['音乐', '歌曲', 'mv']):
            return ContentType.MUSIC
        elif any(word in text for word in ['科技', '技术', '编程']):
            return ContentType.TECHNOLOGY
        elif any(word in text for word in ['新闻', '时事', '报道']):
            return ContentType.NEWS
        else:
            return ContentType.ENTERTAINMENT

    def _analyze_sentiment(self, title: str, description: str) -> Optional[float]:
        """Analyze sentiment of content"""
        text = f"{title} {description}".lower()

        positive_words = ['好', '棒', '优秀', '精彩', '完美', '推荐', '喜欢', '爱']
        negative_words = ['差', '烂', '糟糕', '失望', '讨厌', '垃圾', '无聊']

        pos_count = sum(text.count(word) for word in positive_words)
        neg_count = sum(text.count(word) for word in negative_words)

        if pos_count + neg_count == 0:
            return None

        return (pos_count - neg_count) / (pos_count + neg_count)

    def _generate_content_summary(self, video: VideoInfo) -> str:
        """Generate content summary"""
        title_lower = video.title.lower()

        if '教程' in title_lower:
            summary = f"📚 实用教程：{video.title}"
        elif '音乐' in title_lower:
            summary = f"🎵 音乐内容：{video.title}"
        elif '游戏' in title_lower:
            summary = f"🎮 游戏内容：{video.title}"
        else:
            summary = f"📹 视频内容：{video.title}"

        if video.view_count > 0:
            summary += f" | 已有{self._format_number(video.view_count)}人观看"

        return summary

    def _format_number(self, num: int) -> str:
        """Format large numbers in Chinese style"""
        if num >= 100_000_000:
            return f"{num / 100_000_000:.1f}亿"
        elif num >= 10_000:
            return f"{num / 10_000:.1f}万"
        else:
            return str(num)

    def _video_info_to_dict(self, video: VideoInfo) -> Dict[str, Any]:
        """Convert VideoInfo to dictionary"""
        data = asdict(video)
        data['upload_date'] = video.upload_date.isoformat()
        data['platform'] = video.platform.value
        return data

    def _dict_to_content_summary(self, data: Dict[str, Any]) -> ContentSummary:
        """Convert dictionary to ContentSummary"""
        return ContentSummary(
            ai_summary=data['ai_summary'],
            keywords=data['keywords'],
            sentiment=data.get('sentiment'),
            content_type=ContentType(data.get('content_type', 'other'))
        )

# Export main classes
__all__ = [
    'BilibiliClient',
    'RssGenerator',
    'ContentAnalyzer',
    'VideoInfo',
    'FetchOptions',
    'RssConfig',
    'Platform',
    'TranscriptionStatus',
    'ContentType'
]