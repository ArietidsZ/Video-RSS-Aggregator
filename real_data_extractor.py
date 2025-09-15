#!/usr/bin/env python3
"""
Real Data Extractor - Working with Live APIs
Successfully extracts real data from Chinese video platforms
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import List, Dict, Any
import re
import html
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RealDataExtractor:
    """Extract real data from Chinese platforms with working internet"""

    def __init__(self):
        self.session = None

    async def __aenter__(self):
        # Load Bilibili credentials from environment
        sessdata = os.getenv('BILIBILI_SESSDATA', '')
        bili_jct = os.getenv('BILIBILI_BILI_JCT', '')
        buvid3 = os.getenv('BILIBILI_BUVID3', '')

        # Build cookies for authenticated requests
        cookies = {}
        if sessdata:
            cookies['SESSDATA'] = sessdata
        if bili_jct:
            cookies['bili_jct'] = bili_jct
        if buvid3:
            cookies['buvid3'] = buvid3

        # Create cookie jar if we have credentials
        cookie_jar = None
        if cookies:
            cookie_jar = aiohttp.CookieJar()
            for name, value in cookies.items():
                cookie_jar.update_cookies({name: value})

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/html',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Referer': 'https://www.bilibili.com'
            },
            cookie_jar=cookie_jar
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def extract_bilibili_real_data(self) -> List[Dict[str, Any]]:
        """Extract personalized videos from Bilibili using credentials"""
        videos = []

        # Check if we have credentials
        has_credentials = bool(os.getenv('BILIBILI_SESSDATA'))

        try:
            if has_credentials:
                # Try to fetch personalized content first
                print("Using authenticated Bilibili access for personalized content...")

                # Option 1: User's watch history
                history_url = "https://api.bilibili.com/x/web-interface/history/cursor"
                params = {'ps': 20}  # Page size

                async with self.session.get(history_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == 0:
                            history_items = data.get('data', {}).get('list', [])

                            for item in history_items[:10]:
                                if item.get('history', {}).get('business') == 'archive':
                                    # It's a video
                                    videos.append({
                                        "id": item.get('history', {}).get('bvid', f'bilibili_{len(videos)}'),
                                        "title": item.get('title', ''),
                                        "platform": "bilibili",
                                        "author": item.get('author_name', 'Unknown'),
                                        "url": f"https://www.bilibili.com/video/{item.get('history', {}).get('bvid', '')}",
                                        "description": item.get('new_desc', ''),
                                        "view_count": item.get('stat', {}).get('view', 0),
                                        "upload_date": datetime.now().strftime('%Y-%m-%d'),
                                        "thumbnail": item.get('cover', ''),
                                        "tags": item.get('tag_name', '').split(',') if item.get('tag_name') else [],

                                        # Metadata
                                        "data_source": "bilibili_personalized_history",
                                        "legal_compliance": "Authenticated User Data - Fair Use",
                                        "extraction_method": "Personal watch history - No downloads",
                                        "extracted_at": datetime.now().isoformat()
                                    })

                # Option 2: Recommended videos (personalized feed)
                if len(videos) < 10:
                    recommend_url = "https://api.bilibili.com/x/web-interface/index/top/rcmd"
                    params = {'ps': 10}

                    async with self.session.get(recommend_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('code') == 0:
                                recommendations = data.get('data', {}).get('item', [])

                                for item in recommendations:
                                    if not any(v['id'] == item.get('bvid') for v in videos):
                                        videos.append({
                                            "id": item.get('bvid', f'bilibili_{len(videos)}'),
                                            "title": item.get('title', ''),
                                            "platform": "bilibili",
                                            "author": item.get('owner', {}).get('name', 'Unknown'),
                                            "url": f"https://www.bilibili.com/video/{item.get('bvid', '')}",
                                            "description": item.get('desc', ''),
                                            "view_count": item.get('stat', {}).get('view', 0),
                                            "like_count": item.get('stat', {}).get('like', 0),
                                            "duration": item.get('duration', ''),
                                            "upload_date": datetime.fromtimestamp(item.get('pubdate', 0)).strftime('%Y-%m-%d') if item.get('pubdate') else datetime.now().strftime('%Y-%m-%d'),
                                            "thumbnail": item.get('pic', '').replace('//', 'https://'),
                                            "tags": [],

                                            # Metadata
                                            "data_source": "bilibili_personalized_recommendations",
                                            "legal_compliance": "Authenticated User Data - Fair Use",
                                            "extraction_method": "Personalized recommendations - No downloads",
                                            "extracted_at": datetime.now().isoformat()
                                        })

            # Fallback to public search if no credentials or if personalized fetch failed
            if not videos:
                print("Falling back to public Bilibili search...")
                search_url = "https://api.bilibili.com/x/web-interface/search/all"
                params = {
                    'keyword': '热门',
                    'page': 1,
                    'pagesize': 20
                }

                async with self.session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == 0:
                            results = data.get('data', {}).get('result', {}).get('video', [])

                        for i, item in enumerate(results[:10]):
                            # Clean HTML tags from title
                            title = item.get('title', '').replace('<em class="keyword">', '').replace('</em>', '')
                            title = html.unescape(title)

                            video = {
                                "id": item.get('bvid', f'bilibili_{i}'),
                                "title": title,
                                "platform": "bilibili",
                                "author": item.get('author', 'Unknown'),
                                "url": f"https://www.bilibili.com/video/{item.get('bvid', '')}",
                                "description": item.get('description', ''),
                                "view_count": item.get('play') or 0,
                                "like_count": item.get('favorites') or 0,
                                "duration": item.get('duration', ''),
                                "upload_date": datetime.fromtimestamp(item.get('pubdate', 0)).strftime('%Y-%m-%d') if item.get('pubdate') else datetime.now().strftime('%Y-%m-%d'),
                                "thumbnail": item.get('pic', '').replace('//', 'https://'),
                                "tags": item.get('tag', '').split(',') if item.get('tag') else [],

                                # Legal compliance metadata
                                "data_source": "bilibili_search_api",
                                "legal_compliance": "Public API - Fair Use Academic Research",
                                "extraction_method": "Streaming metadata only - No downloads",
                                "extracted_at": datetime.now().isoformat()
                            }
                            videos.append(video)

        except Exception as e:
            print(f"Bilibili extraction error: {e}")

        return videos

    async def extract_youtube_real_data(self, channel_id: str = "UCBa659QWEk1AI4Tg--mrJ2A") -> List[Dict[str, Any]]:
        """Extract real YouTube data via RSS feeds"""
        videos = []
        try:
            rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

            async with self.session.get(rss_url) as response:
                if response.status == 200:
                    rss_content = await response.text()

                    # Parse video entries from RSS
                    entry_pattern = r'<entry>(.*?)</entry>'
                    entries = re.findall(entry_pattern, rss_content, re.DOTALL)

                    for i, entry in enumerate(entries[:10]):
                        # Extract video data from entry
                        video_id_match = re.search(r'<yt:videoId>(.*?)</yt:videoId>', entry)
                        title_match = re.search(r'<title>(.*?)</title>', entry)
                        author_match = re.search(r'<name>(.*?)</name>', entry)
                        published_match = re.search(r'<published>(.*?)</published>', entry)

                        if all([video_id_match, title_match, author_match, published_match]):
                            video_id = video_id_match.group(1)
                            title = html.unescape(title_match.group(1))
                            author = author_match.group(1)
                            published = published_match.group(1)

                            video = {
                                "id": video_id,
                                "title": title,
                                "platform": "youtube",
                                "author": author,
                                "url": f"https://www.youtube.com/watch?v={video_id}",
                                "description": "Real video from YouTube RSS feed",
                                "upload_date": published.split('T')[0],
                                "thumbnail": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",

                                # Legal compliance metadata
                                "data_source": "youtube_rss_feed",
                                "legal_compliance": "Official RSS - Fair Use Academic Research",
                                "extraction_method": "RSS feed parsing - No downloads",
                                "extracted_at": datetime.now().isoformat()
                            }
                            videos.append(video)

        except Exception as e:
            print(f"YouTube extraction error: {e}")

        return videos

    async def extract_douyin_public_data(self) -> List[Dict[str, Any]]:
        """Attempt to extract Douyin public trending data"""
        videos = []
        try:
            # Try accessing Douyin trending/discover page
            douyin_urls = [
                "https://www.douyin.com/discover",
                "https://www.douyin.com/hot"
            ]

            for url in douyin_urls:
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()

                            # Look for video data in the page source
                            # This is a simplified extraction - real implementation needs more robust parsing
                            video_patterns = [
                                r'"aweme_id":"(\d+)".*?"desc":"([^"]*)".*?"nickname":"([^"]*)"',
                                r'"id":"(\d+)".*?"title":"([^"]*)".*?"author":"([^"]*)"'
                            ]

                            for pattern in video_patterns:
                                matches = re.finditer(pattern, content)
                                for i, match in enumerate(matches):
                                    if i >= 3:  # Limit to 3 videos per pattern
                                        break
                                    if len(videos) >= 5:  # Max 5 total
                                        break

                                    video_id, desc, author = match.groups()

                                    video = {
                                        "id": video_id,
                                        "title": desc[:100] if desc else f"Douyin Video {i+1}",
                                        "platform": "douyin",
                                        "author": author or "Unknown",
                                        "url": f"https://www.douyin.com/video/{video_id}",
                                        "description": desc or "Real content from Douyin",
                                        "upload_date": datetime.now().strftime('%Y-%m-%d'),

                                        # Legal compliance metadata
                                        "data_source": "douyin_public_page",
                                        "legal_compliance": "Public Data - Fair Use Academic Research",
                                        "extraction_method": "Page streaming analysis - No downloads",
                                        "extracted_at": datetime.now().isoformat()
                                    }
                                    videos.append(video)

                            if videos:  # If we found some videos, break
                                break

                except Exception as e:
                    print(f"Error with Douyin URL {url}: {e}")
                    continue

        except Exception as e:
            print(f"Douyin extraction error: {e}")

        return videos

    async def extract_all_real_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Extract real data from all platforms concurrently"""
        print("[START] Extracting REAL data from Chinese video platforms...")
        print("[LEGAL] Legal Framework: Fair Use Academic Research + Official APIs")

        # Run all extractions concurrently
        bilibili_task = self.extract_bilibili_real_data()
        youtube_task = self.extract_youtube_real_data()
        douyin_task = self.extract_douyin_public_data()

        bilibili_videos, youtube_videos, douyin_videos = await asyncio.gather(
            bilibili_task, youtube_task, douyin_task,
            return_exceptions=True
        )

        # Process results
        results = {
            "bilibili": bilibili_videos if not isinstance(bilibili_videos, Exception) else [],
            "youtube": youtube_videos if not isinstance(youtube_videos, Exception) else [],
            "douyin": douyin_videos if not isinstance(douyin_videos, Exception) else [],
            "kuaishou": []  # Would need similar implementation
        }

        # Summary
        total_videos = sum(len(videos) for videos in results.values())
        print(f"\n[STATS] REAL Data Extraction Results:")
        print(f"   Total Videos Extracted: {total_videos}")

        for platform, videos in results.items():
            print(f"   {platform.title()}: {len(videos)} videos")
            if videos and len(videos) > 0:
                print(f"     Sample: {videos[0]['title'][:60]}...")
                print(f"     Author: {videos[0]['author']}")
                print(f"     Legal: {videos[0]['legal_compliance']}")

        return results

# Direct execution
async def main():
    """Extract real data from platforms"""
    async with RealDataExtractor() as extractor:
        real_data = await extractor.extract_all_real_data()

        # Save results
        with open('extracted_real_data.json', 'w', encoding='utf-8') as f:
            json.dump(real_data, f, ensure_ascii=False, indent=2)

        print(f"\n[SUCCESS] Real data extracted and saved to 'extracted_real_data.json'")
        print(f"[METADATA] All extractions include legal compliance metadata")
        print(f"[READY] Ready for academic research use")

        return real_data

if __name__ == "__main__":
    asyncio.run(main())
