"""
Video RSS Aggregator Python SDK

A comprehensive Python SDK for interacting with the Video RSS Aggregator API.
Supports REST, GraphQL, and WebSocket connections.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import aiohttp
import backoff
import jwt
import websockets
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)


# Enums

class ApiVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    PREMIUM = "premium"
    GUEST = "guest"


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


# Models

class Video(BaseModel):
    id: str
    title: str
    description: str
    url: str
    thumbnail_url: Optional[str]
    duration_seconds: int
    channel_id: str
    quality_score: float
    view_count: int
    like_count: int
    created_at: datetime
    updated_at: datetime


class Channel(BaseModel):
    id: str
    name: str
    description: str
    url: str
    subscriber_count: int
    video_count: int
    created_at: datetime


class User(BaseModel):
    id: str
    username: str
    email: str
    role: UserRole
    preferences: Dict[str, Any]
    created_at: datetime
    last_login: Optional[datetime]


class Summary(BaseModel):
    id: str
    video_id: str
    content: str
    key_points: List[str]
    sentiment_score: float
    language: str
    word_count: int
    created_at: datetime


class Recommendation(BaseModel):
    id: str
    user_id: str
    video_id: str
    score: float
    reason: str
    created_at: datetime


class Analytics(BaseModel):
    total_videos: int
    total_channels: int
    total_users: int
    total_summaries: int
    avg_quality_score: float
    processing_rate: float
    cache_hit_rate: float
    error_rate: float


# Pagination

class PageInfo(BaseModel):
    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str]
    end_cursor: Optional[str]


class PaginatedResponse(BaseModel):
    data: List[Any]
    page_info: PageInfo
    total_count: int


# Exceptions

class VideoAggregatorException(Exception):
    """Base exception for SDK"""
    pass


class AuthenticationError(VideoAggregatorException):
    """Authentication failed"""
    pass


class RateLimitError(VideoAggregatorException):
    """Rate limit exceeded"""
    pass


class ApiError(VideoAggregatorException):
    """API returned an error"""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")


# Rate Limiter

class RateLimiter:
    def __init__(self, requests_per_second: int = 10):
        self.requests_per_second = requests_per_second
        self.request_times = []

    async def acquire(self):
        now = time.time()
        # Remove old requests outside the window
        self.request_times = [t for t in self.request_times if now - t < 1.0]

        if len(self.request_times) >= self.requests_per_second:
            sleep_time = 1.0 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                await self.acquire()
        else:
            self.request_times.append(now)


# Main SDK Client

class VideoAggregatorClient:
    """
    Main client for interacting with the Video RSS Aggregator API.

    Example:
        async with VideoAggregatorClient(api_key="your-key") as client:
            videos = await client.get_videos(limit=10)
    """

    def __init__(
        self,
        base_url: str = "https://api.video-aggregator.com",
        api_key: Optional[str] = None,
        api_version: ApiVersion = ApiVersion.V3,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit: int = 100,
        debug: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rate_limit)
        self.debug = debug

        self._session: Optional[aiohttp.ClientSession] = None
        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None

        if debug:
            logging.basicConfig(level=logging.DEBUG)

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    async def _ensure_session(self):
        if not self._session:
            headers = {
                "User-Agent": "VideoAggregatorSDK/1.0 Python",
                "Accept": "application/json",
                "X-API-Version": self.api_version.value,
            }

            if self.api_key:
                headers["X-API-Key"] = self.api_key

            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=self.timeout,
            )

    def _get_url(self, endpoint: str) -> str:
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]

        # Add version to URL if using URL-based versioning
        if self.api_version != ApiVersion.V3:  # V3 is default
            return f"{self.base_url}/api/{self.api_version.value}/{endpoint}"
        return f"{self.base_url}/api/{endpoint}"

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, ApiError),
        max_tries=3,
        max_time=60,
    )
    async def _request(
        self,
        method: HttpMethod,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request with rate limiting and retries."""
        await self._ensure_session()
        await self.rate_limiter.acquire()

        url = self._get_url(endpoint)

        # Merge headers
        req_headers = headers or {}
        if self._token and self._token_expires and self._token_expires > datetime.utcnow():
            req_headers["Authorization"] = f"Bearer {self._token}"

        logger.debug(f"Request: {method} {url}")

        async with self._session.request(
            method.value,
            url,
            params=params,
            json=json_data,
            headers=req_headers,
        ) as response:
            # Handle rate limiting
            if response.status == 429:
                retry_after = response.headers.get("Retry-After", "60")
                raise RateLimitError(f"Rate limited. Retry after {retry_after} seconds")

            # Handle authentication errors
            if response.status == 401:
                raise AuthenticationError("Authentication failed")

            # Handle other errors
            if response.status >= 400:
                error_data = await response.json()
                raise ApiError(response.status, error_data.get("message", "Unknown error"))

            # Check for deprecation warnings
            if "X-API-Deprecated" in response.headers:
                logger.warning(f"API version {self.api_version} is deprecated")

            return await response.json()

    # Authentication

    async def authenticate(self, username: str, password: str) -> str:
        """Authenticate and get access token."""
        response = await self._request(
            HttpMethod.POST,
            "auth/login",
            json_data={"username": username, "password": password},
        )

        self._token = response["access_token"]
        self._token_expires = datetime.utcnow() + timedelta(seconds=response.get("expires_in", 3600))

        return self._token

    async def refresh_token(self, refresh_token: str) -> str:
        """Refresh access token."""
        response = await self._request(
            HttpMethod.POST,
            "auth/refresh",
            json_data={"refresh_token": refresh_token},
        )

        self._token = response["access_token"]
        self._token_expires = datetime.utcnow() + timedelta(seconds=response.get("expires_in", 3600))

        return self._token

    # Video operations

    async def get_videos(
        self,
        limit: int = 20,
        offset: int = 0,
        channel_id: Optional[str] = None,
        min_quality_score: Optional[float] = None,
        search: Optional[str] = None,
    ) -> PaginatedResponse:
        """Get paginated list of videos."""
        params = {
            "limit": limit,
            "offset": offset,
        }

        if channel_id:
            params["channel_id"] = channel_id
        if min_quality_score:
            params["min_quality_score"] = min_quality_score
        if search:
            params["search"] = search

        response = await self._request(HttpMethod.GET, "videos", params=params)

        return PaginatedResponse(
            data=[Video(**v) for v in response["data"]],
            page_info=PageInfo(**response["page_info"]),
            total_count=response["total_count"],
        )

    async def get_video(self, video_id: str) -> Video:
        """Get single video by ID."""
        response = await self._request(HttpMethod.GET, f"videos/{video_id}")
        return Video(**response)

    async def create_video(
        self,
        title: str,
        description: str,
        url: str,
        channel_id: str,
        duration_seconds: int,
        thumbnail_url: Optional[str] = None,
    ) -> Video:
        """Create a new video."""
        data = {
            "title": title,
            "description": description,
            "url": url,
            "channel_id": channel_id,
            "duration_seconds": duration_seconds,
        }

        if thumbnail_url:
            data["thumbnail_url"] = thumbnail_url

        response = await self._request(HttpMethod.POST, "videos", json_data=data)
        return Video(**response)

    async def update_video(
        self,
        video_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> Video:
        """Update video details."""
        data = {}
        if title:
            data["title"] = title
        if description:
            data["description"] = description
        if quality_score:
            data["quality_score"] = quality_score

        response = await self._request(
            HttpMethod.PATCH, f"videos/{video_id}", json_data=data
        )
        return Video(**response)

    async def delete_video(self, video_id: str) -> bool:
        """Delete a video."""
        await self._request(HttpMethod.DELETE, f"videos/{video_id}")
        return True

    # Channel operations

    async def get_channels(self) -> List[Channel]:
        """Get all channels."""
        response = await self._request(HttpMethod.GET, "channels")
        return [Channel(**c) for c in response]

    async def get_channel(self, channel_id: str) -> Channel:
        """Get single channel by ID."""
        response = await self._request(HttpMethod.GET, f"channels/{channel_id}")
        return Channel(**response)

    # Summary operations

    async def get_summary(self, video_id: str) -> Summary:
        """Get summary for a video."""
        response = await self._request(HttpMethod.GET, f"videos/{video_id}/summary")
        return Summary(**response)

    async def create_summary(
        self,
        video_id: str,
        content: str,
        key_points: List[str],
    ) -> Summary:
        """Create video summary."""
        data = {
            "video_id": video_id,
            "content": content,
            "key_points": key_points,
        }

        response = await self._request(HttpMethod.POST, "summaries", json_data=data)
        return Summary(**response)

    # Recommendation operations

    async def get_recommendations(
        self,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Recommendation]:
        """Get recommendations for a user."""
        params = {"limit": limit}
        if user_id:
            params["user_id"] = user_id

        response = await self._request(HttpMethod.GET, "recommendations", params=params)
        return [Recommendation(**r) for r in response]

    # Analytics

    async def get_analytics(self) -> Analytics:
        """Get system analytics."""
        response = await self._request(HttpMethod.GET, "analytics")
        return Analytics(**response)

    # GraphQL support

    async def graphql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute GraphQL query."""
        data = {"query": query}
        if variables:
            data["variables"] = variables

        return await self._request(HttpMethod.POST, "graphql", json_data=data)

    # WebSocket support

    async def subscribe(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], None],
    ):
        """Subscribe to real-time events via WebSocket."""
        ws_url = self.base_url.replace("http", "ws") + "/ws"

        async with websockets.connect(ws_url) as websocket:
            # Send subscription
            await websocket.send(json.dumps({
                "type": "subscribe",
                "event": event_type,
                "auth": self._token,
            }))

            # Listen for events
            async for message in websocket:
                data = json.loads(message)
                callback(data)

    # Webhook signature verification

    @staticmethod
    def verify_webhook_signature(
        payload: bytes,
        signature: str,
        secret: str,
    ) -> bool:
        """Verify webhook signature."""
        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)

    # Batch operations

    async def batch_request(
        self,
        operations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute multiple operations in a single request."""
        response = await self._request(
            HttpMethod.POST,
            "batch",
            json_data={"operations": operations},
        )
        return response["results"]


# Async context manager usage example

async def example_usage():
    async with VideoAggregatorClient(api_key="your-api-key") as client:
        # Get videos
        videos = await client.get_videos(limit=10)
        print(f"Found {videos.total_count} videos")

        for video in videos.data:
            print(f"- {video.title} (Score: {video.quality_score})")

        # Get recommendations
        recommendations = await client.get_recommendations()
        for rec in recommendations:
            print(f"Recommended: {rec.video_id} (Score: {rec.score})")

        # GraphQL query
        result = await client.graphql("""
            query {
                videos(limit: 5) {
                    edges {
                        node {
                            id
                            title
                            channel {
                                name
                            }
                        }
                    }
                }
            }
        """)

        # Subscribe to events
        def on_video_created(data):
            print(f"New video: {data}")

        await client.subscribe("video.created", on_video_created)


if __name__ == "__main__":
    asyncio.run(example_usage())