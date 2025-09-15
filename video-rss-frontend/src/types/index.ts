export interface Video {
  id: string;
  title: string;
  description?: string;
  url: string;
  author: string;
  platform: 'bilibili' | 'douyin' | 'kuaishou';
  upload_date: string;
  duration?: number;
  view_count: number;
  like_count: number;
  comment_count: number;
  thumbnail_url?: string;
  tags: string[];
  transcription?: Transcription;
  created_at: string;
  updated_at: string;
}

export interface Transcription {
  id: number;
  video_id: string;
  paragraph_summary?: string;
  sentence_subtitle?: string;
  full_transcript?: string;
  status: 'success' | 'pending' | 'failed' | 'unavailable';
  transcriber_model: string;
  summarizer_model: string;
  processing_time_ms?: number;
  confidence_score?: number;
  language_detected?: string;
  created_at: string;
  updated_at: string;
}

export interface ApiStats {
  total_videos: number;
  videos_by_platform: Record<string, number>;
  requests_per_minute: number;
  avg_response_time_ms: number;
  cache_hit_rate: number;
  transcription_queue_size: number;
  system_metrics: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
  };
  platform_status: Record<string, boolean>;
}

export interface Config {
  platforms: {
    bilibili: PlatformConfig;
    douyin: PlatformConfig;
    kuaishou: PlatformConfig;
  };
  rss: RssConfig;
  transcription: TranscriptionConfig;
  cache: CacheConfig;
  rate_limiting: RateLimitConfig;
}

export interface PlatformConfig {
  enabled: boolean;
  rate_limit_per_minute: number;
  max_videos_per_request: number;
  retry_attempts: number;
  timeout_seconds: number;
}

export interface RssConfig {
  title: string;
  description: string;
  max_items: number;
  cache_ttl_seconds: number;
  include_transcription: boolean;
}

export interface TranscriptionConfig {
  enabled: boolean;
  model: string;
  language: string;
  max_concurrent: number;
  timeout_seconds: number;
}

export interface CacheConfig {
  redis_enabled: boolean;
  redis_url: string;
  memory_cache_size: number;
  default_ttl_seconds: number;
}

export interface RateLimitConfig {
  requests_per_second: number;
  burst_capacity: number;
  enable_per_ip_limiting: boolean;
}

export interface VideoFilters {
  platforms?: string[];
  search?: string;
  sort_by?: 'upload_date' | 'view_count' | 'like_count';
  limit?: number;
  offset?: number;
}

export interface ApiError {
  error: string;
  message: string;
  status: number;
}