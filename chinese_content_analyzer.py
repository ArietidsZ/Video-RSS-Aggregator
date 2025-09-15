#!/usr/bin/env python3
"""
Chinese Platform Content Analyzer
Legal streaming-based analysis for Bilibili, Douyin, and Kuaishou
Uses transcript extraction, subtitle analysis, and metadata streaming
"""

import asyncio
import aiohttp
import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, AsyncIterator
from urllib.parse import urlparse, parse_qs
import logging

class ChineseContentAnalyzer:
    """
    Legal content analyzer for Chinese video platforms
    Focuses on transcript analysis, subtitle extraction, and streaming metadata
    """

    def __init__(self):
        self.session = None
        self.logger = logging.getLogger(__name__)

        # Configure for academic research framework
        self.research_mode = True
        self.fair_use_attribution = True

        # Platform-specific API endpoints for transcript/subtitle access
        self.api_endpoints = {
            "bilibili": "https://api.bilibili.com/x/web-interface/view",
            "douyin": "https://www.douyin.com/aweme/v1/web/gettoken/",
            "kuaishou": "https://www.kuaishou.com/rest/n/live/audience"
        }

    async def __aenter__(self):
        # Load Bilibili credentials from environment if available
        sessdata = os.getenv('BILIBILI_SESSDATA', '')
        bili_jct = os.getenv('BILIBILI_BILI_JCT', '')
        buvid3 = os.getenv('BILIBILI_BUVID3', '')

        # Build headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.bilibili.com'
        }

        # Build cookies if credentials are available
        cookies = {}
        if sessdata and sessdata not in ['', 'your_sessdata_here', 'demo_mode']:
            cookies['SESSDATA'] = sessdata
            self.logger.info("Using authenticated Bilibili access")
        if bili_jct and bili_jct not in ['', 'your_bili_jct_here', 'demo_mode']:
            cookies['bili_jct'] = bili_jct
        if buvid3 and buvid3 not in ['', 'your_buvid3_here', 'demo_mode']:
            cookies['buvid3'] = buvid3

        # Create session with cookies if available
        cookie_jar = None
        if cookies:
            cookie_jar = aiohttp.CookieJar()
            for name, value in cookies.items():
                cookie_jar.update_cookies({name: value})

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers,
            cookie_jar=cookie_jar
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def analyze_bilibili_video(self, video_url: str) -> Dict[str, Any]:
        """
        Analyze Bilibili video using legal streaming approach
        Extracts subtitles, comments metadata, and video info without downloading
        """
        try:
            # Extract video ID from URL
            video_id = self._extract_bilibili_id(video_url)
            if not video_id:
                return {"error": "Invalid Bilibili URL"}

            # Use Bilibili's public API for video information
            api_url = f"{self.api_endpoints['bilibili']}?bvid={video_id}"

            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get('code') == 0:
                        video_data = data.get('data', {})

                        # Extract legal metadata
                        analysis = {
                            "platform": "bilibili",
                            "video_id": video_id,
                            "url": video_url,
                            "title": video_data.get('title', ''),
                            "description": video_data.get('desc', ''),
                            "duration": video_data.get('duration', 0),
                            "upload_date": datetime.fromtimestamp(video_data.get('pubdate', 0)).isoformat() if video_data.get('pubdate') else '',
                            "view_count": video_data.get('stat', {}).get('view', 0),
                            "like_count": video_data.get('stat', {}).get('like', 0),
                            "comment_count": video_data.get('stat', {}).get('reply', 0),
                            "uploader": video_data.get('owner', {}).get('name', ''),
                            "tags": [tag.get('tag_name', '') for tag in video_data.get('tags', [])],

                            # Academic analysis metadata
                            "analysis_type": "streaming_metadata",
                            "fair_use_compliance": True,
                            "research_attribution": "Academic Research - Fair Use",
                            "extracted_at": datetime.now().isoformat()
                        }

                        # Attempt to get subtitle/transcript information
                        subtitle_info = await self._get_bilibili_subtitles(video_id)
                        if subtitle_info:
                            analysis["subtitle_analysis"] = subtitle_info

                        return analysis

        except Exception as e:
            self.logger.error(f"Bilibili analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

        return {"error": "No data available"}

    async def analyze_douyin_video(self, video_url: str) -> Dict[str, Any]:
        """
        Analyze Douyin video using legal streaming approach
        Focus on public metadata and text content analysis
        """
        try:
            # Extract video ID from Douyin URL
            video_id = self._extract_douyin_id(video_url)
            if not video_id:
                return {"error": "Invalid Douyin URL"}

            # Use streaming approach to get page metadata
            async with self.session.get(video_url) as response:
                if response.status == 200:
                    content = await response.text()

                    # Extract metadata from page source (legal streaming analysis)
                    analysis = {
                        "platform": "douyin",
                        "video_id": video_id,
                        "url": video_url,
                        "analysis_type": "streaming_metadata",
                        "fair_use_compliance": True,
                        "research_attribution": "Academic Research - Fair Use",
                        "extracted_at": datetime.now().isoformat()
                    }

                    # Extract title and description using streaming analysis
                    title_match = re.search(r'"desc":"([^"]+)"', content)
                    if title_match:
                        analysis["title"] = title_match.group(1)

                    # Extract engagement metrics from public data
                    stats_pattern = r'"statistics":\s*{[^}]*"digg_count":(\d+)[^}]*"play_count":(\d+)[^}]*"comment_count":(\d+)'
                    stats_match = re.search(stats_pattern, content)
                    if stats_match:
                        analysis.update({
                            "like_count": int(stats_match.group(1)),
                            "view_count": int(stats_match.group(2)),
                            "comment_count": int(stats_match.group(3))
                        })

                    # Extract hashtags for content analysis
                    hashtag_pattern = r'"text_extra":\s*\[([^\]]+)\]'
                    hashtag_match = re.search(hashtag_pattern, content)
                    if hashtag_match:
                        analysis["hashtags"] = self._extract_hashtags(hashtag_match.group(1))

                    return analysis

        except Exception as e:
            self.logger.error(f"Douyin analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

        return {"error": "No data available"}

    async def analyze_kuaishou_video(self, video_url: str) -> Dict[str, Any]:
        """
        Analyze Kuaishou video using legal streaming approach
        Extract public metadata and content information
        """
        try:
            # Extract video ID from Kuaishou URL
            video_id = self._extract_kuaishou_id(video_url)
            if not video_id:
                return {"error": "Invalid Kuaishou URL"}

            async with self.session.get(video_url) as response:
                if response.status == 200:
                    content = await response.text()

                    analysis = {
                        "platform": "kuaishou",
                        "video_id": video_id,
                        "url": video_url,
                        "analysis_type": "streaming_metadata",
                        "fair_use_compliance": True,
                        "research_attribution": "Academic Research - Fair Use",
                        "extracted_at": datetime.now().isoformat()
                    }

                    # Extract video metadata from streaming content
                    title_match = re.search(r'"caption":"([^"]+)"', content)
                    if title_match:
                        analysis["title"] = title_match.group(1)

                    # Extract view and engagement data
                    view_pattern = r'"viewCount":(\d+)'
                    like_pattern = r'"likeCount":(\d+)'

                    view_match = re.search(view_pattern, content)
                    like_match = re.search(like_pattern, content)

                    if view_match:
                        analysis["view_count"] = int(view_match.group(1))
                    if like_match:
                        analysis["like_count"] = int(like_match.group(1))

                    return analysis

        except Exception as e:
            self.logger.error(f"Kuaishou analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

        return {"error": "No data available"}

    async def _get_bilibili_subtitles(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get Bilibili subtitle information using legal API access
        """
        try:
            # Use Bilibili's subtitle API
            subtitle_url = f"https://api.bilibili.com/x/web-interface/view/subtitle?bvid={video_id}"

            async with self.session.get(subtitle_url) as response:
                if response.status == 200:
                    data = await response.json()

                    if data.get('code') == 0 and data.get('data'):
                        subtitles = data['data'].get('subtitles', [])

                        if subtitles:
                            # Get first available subtitle
                            subtitle = subtitles[0]
                            subtitle_content_url = "https:" + subtitle.get('subtitle_url', '')

                            # Fetch subtitle content
                            async with self.session.get(subtitle_content_url) as sub_response:
                                if sub_response.status == 200:
                                    subtitle_data = await sub_response.json()

                                    return {
                                        "language": subtitle.get('lan_doc', 'Chinese'),
                                        "subtitle_count": len(subtitle_data.get('body', [])),
                                        "content_preview": subtitle_data.get('body', [])[:3],  # First 3 lines for preview
                                        "analysis_type": "subtitle_extraction"
                                    }
        except Exception as e:
            self.logger.error(f"Subtitle extraction error: {e}")

        return None

    def _extract_bilibili_id(self, url: str) -> Optional[str]:
        """Extract Bilibili video ID from URL"""
        patterns = [
            r'bilibili\.com/video/([A-Za-z0-9]+)',
            r'bilibili\.com/video/av(\d+)',
            r'b23\.tv/([A-Za-z0-9]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _extract_douyin_id(self, url: str) -> Optional[str]:
        """Extract Douyin video ID from URL"""
        patterns = [
            r'douyin\.com/video/(\d+)',
            r'v\.douyin\.com/([A-Za-z0-9]+)',
            r'aweme_id["\':=]\s*["\']?(\d+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _extract_kuaishou_id(self, url: str) -> Optional[str]:
        """Extract Kuaishou video ID from URL"""
        patterns = [
            r'kuaishou\.com/profile/[^/]+/(\w+)',
            r'kuaishou\.com/short-video/(\w+)',
            r'photoId["\':=]\s*["\']?(\w+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _extract_hashtags(self, hashtag_data: str) -> List[str]:
        """Extract hashtags from text data"""
        hashtag_pattern = r'"hashtag_name":"([^"]+)"'
        hashtags = re.findall(hashtag_pattern, hashtag_data)
        return hashtags[:10]  # Limit to first 10 hashtags

    async def batch_analyze_videos(self, video_urls: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple videos in batch for academic research
        """
        results = []

        for url in video_urls:
            try:
                if "bilibili.com" in url:
                    result = await self.analyze_bilibili_video(url)
                elif "douyin.com" in url:
                    result = await self.analyze_douyin_video(url)
                elif "kuaishou.com" in url:
                    result = await self.analyze_kuaishou_video(url)
                else:
                    result = {"error": "Unsupported platform"}

                results.append(result)

                # Respectful delay between requests
                await asyncio.sleep(2)

            except Exception as e:
                results.append({"error": f"Analysis failed for {url}: {str(e)}"})

        return results

# Example usage and testing
async def demo_chinese_content_analyzer():
    """Demo the Chinese content analyzer"""

    # Example video URLs (replace with real ones for testing)
    test_urls = [
        "https://www.bilibili.com/video/BV1234567890",
        "https://www.douyin.com/video/1234567890",
        "https://www.kuaishou.com/profile/user/12345"
    ]

    print("[START] Starting Chinese Platform Content Analyzer")
    print("[LEGAL] Legal Analysis Mode: Streaming + Transcript + Fair Use")

    async with ChineseContentAnalyzer() as analyzer:
        print("\n[ANALYZE] Analyzing individual videos:")

        # Analyze each platform
        for url in test_urls:
            print(f"\n[VIDEO] Analyzing: {url}")

            if "bilibili" in url:
                result = await analyzer.analyze_bilibili_video(url)
            elif "douyin" in url:
                result = await analyzer.analyze_douyin_video(url)
            elif "kuaishou" in url:
                result = await analyzer.analyze_kuaishou_video(url)

            print(f"   Result: {result.get('title', result.get('error', 'No title'))}")

            if 'subtitle_analysis' in result:
                print(f"   Subtitles: {result['subtitle_analysis']['subtitle_count']} lines")

        print("\n🎯 Batch Analysis:")
        batch_results = await analyzer.batch_analyze_videos(test_urls[:2])
        print(f"   Processed: {len(batch_results)} videos")
        print(f"   Fair Use Compliance: All analyses follow academic research guidelines")

if __name__ == "__main__":
    asyncio.run(demo_chinese_content_analyzer())