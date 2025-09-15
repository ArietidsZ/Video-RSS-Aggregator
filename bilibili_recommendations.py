#!/usr/bin/env python3
"""
Bilibili Personalized Recommendations Fetcher
Fetches actual personalized recommendations from Bilibili using user credentials
"""

import asyncio
import aiohttp
import json
import os
from datetime import datetime
from typing import List, Dict, Any

class BilibiliRecommendationFetcher:
    """Fetch personalized recommendations from Bilibili platform"""

    def __init__(self):
        self.session = None

    async def __aenter__(self):
        # Load credentials from environment
        self.sessdata = os.getenv('BILIBILI_SESSDATA', '')
        self.bili_jct = os.getenv('BILIBILI_BILI_JCT', '')
        self.buvid3 = os.getenv('BILIBILI_BUVID3', '')

        # Set up authenticated session
        cookies = {
            'SESSDATA': self.sessdata,
            'bili_jct': self.bili_jct,
            'buvid3': self.buvid3
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.bilibili.com',
            'Cookie': '; '.join([f'{k}={v}' for k, v in cookies.items() if v])
        }

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers,
            cookies=cookies if any(cookies.values()) else None
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_recommendations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch personalized recommendations from Bilibili recommendation API"""
        videos = []

        try:
            # Bilibili recommendation API endpoint
            recommend_url = "https://api.bilibili.com/x/web-interface/index/top/rcmd"
            params = {
                'ps': limit,  # page size
                'fresh_type': 3,  # fresh recommendations
                'feed_version': 'V8'  # current feed version
            }

            print(f"[INFO] Fetching {limit} personalized recommendations from Bilibili...")

            async with self.session.get(recommend_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('code') == 0:
                        items = data.get('data', {}).get('item', [])
                        print(f"[SUCCESS] Found {len(items)} personalized recommendations")

                        for item in items[:limit]:
                            # Extract video information
                            video_info = {
                                "id": item.get('bvid', item.get('id', '')),
                                "title": item.get('title', ''),
                                "platform": "bilibili",
                                "author": item.get('owner', {}).get('name', 'Unknown'),
                                "url": f"https://www.bilibili.com/video/{item.get('bvid', item.get('id', ''))}",
                                "description": item.get('desc', ''),
                                "view_count": item.get('stat', {}).get('view', 0),
                                "like_count": item.get('stat', {}).get('like', 0),
                                "duration": self._format_duration(item.get('duration', 0)),
                                "upload_date": datetime.fromtimestamp(item.get('pubdate', 0)).strftime('%Y-%m-%d') if item.get('pubdate') else datetime.now().strftime('%Y-%m-%d'),
                                "thumbnail": item.get('pic', '').replace('http://', 'https://'),
                                "tags": [tag.get('tag_name', '') for tag in item.get('tag', [])],
                                "reason": item.get('rcmd_reason', {}).get('content', ''),  # Why this was recommended

                                # Metadata
                                "data_source": "bilibili_recommendation_api",
                                "legal_compliance": "Authenticated API - Personal Recommendations",
                                "extraction_method": "Official recommendation feed",
                                "extracted_at": datetime.now().isoformat()
                            }
                            videos.append(video_info)
                    else:
                        print(f"[ERROR] Bilibili API error: {data.get('message', 'Unknown error')}")

                        # Fallback to feed API
                        return await self._fetch_feed_recommendations(limit)
                else:
                    print(f"[ERROR] HTTP {response.status} from Bilibili recommendation API")
                    return await self._fetch_feed_recommendations(limit)

        except Exception as e:
            print(f"[ERROR] Recommendation fetch failed: {e}")
            return await self._fetch_feed_recommendations(limit)

        return videos

    async def _fetch_feed_recommendations(self, limit: int) -> List[Dict[str, Any]]:
        """Fallback: Fetch from user's personalized feed"""
        videos = []

        try:
            feed_url = "https://api.bilibili.com/x/web-interface/wbi/index/top/feed/rcmd"
            params = {
                'ps': limit,
                'fresh_idx': 1,
                'fresh_idx_1h': 1,
                'brush': 1,
                'homepage_ver': 1
            }

            print(f"[FALLBACK] Using personalized feed API...")

            async with self.session.get(feed_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('code') == 0:
                        items = data.get('data', {}).get('item', [])

                        for item in items[:limit]:
                            video_info = {
                                "id": item.get('bvid', ''),
                                "title": item.get('title', ''),
                                "platform": "bilibili",
                                "author": item.get('owner', {}).get('name', 'Unknown'),
                                "url": f"https://www.bilibili.com/video/{item.get('bvid', '')}",
                                "description": item.get('desc', ''),
                                "view_count": item.get('stat', {}).get('view', 0),
                                "like_count": item.get('stat', {}).get('like', 0),
                                "duration": self._format_duration(item.get('duration', 0)),
                                "upload_date": datetime.fromtimestamp(item.get('pubdate', 0)).strftime('%Y-%m-%d') if item.get('pubdate') else datetime.now().strftime('%Y-%m-%d'),
                                "thumbnail": item.get('pic', '').replace('http://', 'https://'),
                                "tags": [tag.get('tag_name', '') for tag in item.get('tag', [])],

                                "data_source": "bilibili_feed_api",
                                "legal_compliance": "Authenticated Feed - Personal Recommendations",
                                "extraction_method": "Personal feed API",
                                "extracted_at": datetime.now().isoformat()
                            }
                            videos.append(video_info)

        except Exception as e:
            print(f"[ERROR] Feed fallback failed: {e}")

        return videos

    def _format_duration(self, duration_seconds: int) -> str:
        """Convert duration from seconds to MM:SS or HH:MM:SS format"""
        if not duration_seconds:
            return "0:00"

        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

# Test function
async def test_recommendations():
    """Test the recommendation fetcher"""
    async with BilibiliRecommendationFetcher() as fetcher:
        recommendations = await fetcher.fetch_recommendations(10)

        print(f"\n[RESULTS] Fetched {len(recommendations)} personalized recommendations:")
        for i, video in enumerate(recommendations[:5], 1):
            try:
                title = video['title'].encode('utf-8', errors='replace').decode('utf-8')
                author = video['author'].encode('utf-8', errors='replace').decode('utf-8')
                print(f"{i}. {title}")
                print(f"   Author: {author} | Views: {video['view_count']:,}")
                if video.get('reason'):
                    reason = video['reason'].encode('utf-8', errors='replace').decode('utf-8')
                    print(f"   Recommendation reason: {reason}")
            except UnicodeEncodeError:
                print(f"{i}. [Chinese video title - {len(video['title'])} chars]")
                print(f"   Author: [Chinese name] | Views: {video['view_count']:,}")
            print()

if __name__ == "__main__":
    asyncio.run(test_recommendations())