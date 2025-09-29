/**
 * Video RSS Aggregator TypeScript SDK
 *
 * A comprehensive TypeScript SDK for interacting with the Video RSS Aggregator API.
 * Supports REST, GraphQL, and WebSocket connections.
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { EventEmitter } from 'events';
import WebSocket from 'ws';
import * as jwt from 'jsonwebtoken';
import * as crypto from 'crypto';

// Enums

export enum ApiVersion {
  V1 = 'v1',
  V2 = 'v2',
  V3 = 'v3',
}

export enum UserRole {
  ADMIN = 'admin',
  USER = 'user',
  PREMIUM = 'premium',
  GUEST = 'guest',
}

export enum HttpMethod {
  GET = 'GET',
  POST = 'POST',
  PUT = 'PUT',
  DELETE = 'DELETE',
  PATCH = 'PATCH',
}

// Interfaces and Types

export interface Video {
  id: string;
  title: string;
  description: string;
  url: string;
  thumbnailUrl?: string;
  durationSeconds: number;
  channelId: string;
  qualityScore: number;
  viewCount: number;
  likeCount: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface Channel {
  id: string;
  name: string;
  description: string;
  url: string;
  subscriberCount: number;
  videoCount: number;
  createdAt: Date;
}

export interface User {
  id: string;
  username: string;
  email: string;
  role: UserRole;
  preferences: UserPreferences;
  createdAt: Date;
  lastLogin?: Date;
}

export interface UserPreferences {
  categories: string[];
  languages: string[];
  minQualityScore: number;
  maxDurationSeconds: number;
  notificationsEnabled: boolean;
}

export interface Summary {
  id: string;
  videoId: string;
  content: string;
  keyPoints: string[];
  sentimentScore: number;
  language: string;
  wordCount: number;
  createdAt: Date;
}

export interface Recommendation {
  id: string;
  userId: string;
  videoId: string;
  score: number;
  reason: string;
  createdAt: Date;
}

export interface Analytics {
  totalVideos: number;
  totalChannels: number;
  totalUsers: number;
  totalSummaries: number;
  avgQualityScore: number;
  processingRate: number;
  cacheHitRate: number;
  errorRate: number;
}

export interface PageInfo {
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  startCursor?: string;
  endCursor?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pageInfo: PageInfo;
  totalCount: number;
}

export interface VideoFilter {
  channelId?: string;
  minQualityScore?: number;
  maxDurationSeconds?: number;
  category?: string;
  language?: string;
  search?: string;
}

export interface PaginationOptions {
  limit?: number;
  offset?: number;
  cursor?: string;
}

export interface ClientConfig {
  baseUrl?: string;
  apiKey?: string;
  apiVersion?: ApiVersion;
  timeout?: number;
  maxRetries?: number;
  rateLimit?: number;
  debug?: boolean;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken?: string;
  expiresIn?: number;
}

export interface BatchOperation {
  method: HttpMethod;
  endpoint: string;
  params?: Record<string, any>;
  data?: Record<string, any>;
}

export interface WebhookEvent {
  id: string;
  type: string;
  timestamp: Date;
  data: any;
  signature?: string;
}

// Custom Errors

export class VideoAggregatorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'VideoAggregatorError';
  }
}

export class AuthenticationError extends VideoAggregatorError {
  constructor(message: string = 'Authentication failed') {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends VideoAggregatorError {
  retryAfter: number;

  constructor(retryAfter: number) {
    super(`Rate limit exceeded. Retry after ${retryAfter} seconds`);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class ApiError extends VideoAggregatorError {
  statusCode: number;

  constructor(statusCode: number, message: string) {
    super(`API Error ${statusCode}: ${message}`);
    this.name = 'ApiError';
    this.statusCode = statusCode;
  }
}

// Rate Limiter

class RateLimiter {
  private requestsPerSecond: number;
  private requestTimes: number[] = [];

  constructor(requestsPerSecond: number) {
    this.requestsPerSecond = requestsPerSecond;
  }

  async acquire(): Promise<void> {
    const now = Date.now();
    // Remove old requests outside the window
    this.requestTimes = this.requestTimes.filter(t => now - t < 1000);

    if (this.requestTimes.length >= this.requestsPerSecond) {
      const sleepTime = 1000 - (now - this.requestTimes[0]);
      if (sleepTime > 0) {
        await new Promise(resolve => setTimeout(resolve, sleepTime));
        return this.acquire();
      }
    }

    this.requestTimes.push(now);
  }
}

// Main SDK Client

export class VideoAggregatorClient extends EventEmitter {
  private axios: AxiosInstance;
  private config: Required<ClientConfig>;
  private rateLimiter: RateLimiter;
  private accessToken?: string;
  private tokenExpires?: Date;
  private ws?: WebSocket;

  constructor(config: ClientConfig = {}) {
    super();

    this.config = {
      baseUrl: config.baseUrl || 'https://api.video-aggregator.com',
      apiKey: config.apiKey || '',
      apiVersion: config.apiVersion || ApiVersion.V3,
      timeout: config.timeout || 30000,
      maxRetries: config.maxRetries || 3,
      rateLimit: config.rateLimit || 100,
      debug: config.debug || false,
    };

    this.rateLimiter = new RateLimiter(this.config.rateLimit);

    // Setup axios instance
    this.axios = axios.create({
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      headers: {
        'User-Agent': 'VideoAggregatorSDK/1.0 TypeScript',
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'X-API-Version': this.config.apiVersion,
      },
    });

    if (this.config.apiKey) {
      this.axios.defaults.headers.common['X-API-Key'] = this.config.apiKey;
    }

    // Add request interceptor for auth and rate limiting
    this.axios.interceptors.request.use(
      async (config) => {
        await this.rateLimiter.acquire();

        if (this.accessToken && this.tokenExpires && this.tokenExpires > new Date()) {
          config.headers!.Authorization = `Bearer ${this.accessToken}`;
        }

        if (this.config.debug) {
          console.log(`Request: ${config.method?.toUpperCase()} ${config.url}`);
        }

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Add response interceptor for error handling and retries
    this.axios.interceptors.response.use(
      (response) => {
        // Check for deprecation warnings
        if (response.headers['x-api-deprecated']) {
          console.warn(`API version ${this.config.apiVersion} is deprecated`);
          this.emit('deprecation', {
            version: this.config.apiVersion,
            message: response.headers['x-api-deprecation-message'],
          });
        }

        return response;
      },
      async (error) => {
        if (error.response) {
          const { status, data } = error.response;

          if (status === 429) {
            const retryAfter = parseInt(error.response.headers['retry-after'] || '60');
            throw new RateLimitError(retryAfter);
          }

          if (status === 401) {
            throw new AuthenticationError(data.message);
          }

          throw new ApiError(status, data.message || 'Unknown error');
        }

        throw error;
      }
    );
  }

  private getUrl(endpoint: string): string {
    if (endpoint.startsWith('/')) {
      endpoint = endpoint.substring(1);
    }

    // Add version to URL if using URL-based versioning
    if (this.config.apiVersion !== ApiVersion.V3) {
      return `/api/${this.config.apiVersion}/${endpoint}`;
    }
    return `/api/${endpoint}`;
  }

  private async request<T>(
    method: HttpMethod,
    endpoint: string,
    params?: Record<string, any>,
    data?: Record<string, any>
  ): Promise<T> {
    const url = this.getUrl(endpoint);

    const config: AxiosRequestConfig = {
      method,
      url,
      params,
      data,
    };

    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.config.maxRetries; attempt++) {
      try {
        const response = await this.axios.request<T>(config);
        return response.data;
      } catch (error) {
        lastError = error as Error;

        // Don't retry on client errors (4xx)
        if (error instanceof ApiError && error.statusCode >= 400 && error.statusCode < 500) {
          throw error;
        }

        // Exponential backoff
        if (attempt < this.config.maxRetries - 1) {
          const delay = Math.pow(2, attempt) * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError || new Error('Request failed');
  }

  // Authentication

  async authenticate(username: string, password: string): Promise<AuthTokens> {
    const response = await this.request<AuthTokens>(
      HttpMethod.POST,
      'auth/login',
      undefined,
      { username, password }
    );

    this.accessToken = response.accessToken;
    if (response.expiresIn) {
      this.tokenExpires = new Date(Date.now() + response.expiresIn * 1000);
    }

    return response;
  }

  async refreshToken(refreshToken: string): Promise<AuthTokens> {
    const response = await this.request<AuthTokens>(
      HttpMethod.POST,
      'auth/refresh',
      undefined,
      { refreshToken }
    );

    this.accessToken = response.accessToken;
    if (response.expiresIn) {
      this.tokenExpires = new Date(Date.now() + response.expiresIn * 1000);
    }

    return response;
  }

  setAccessToken(token: string, expiresIn?: number): void {
    this.accessToken = token;
    if (expiresIn) {
      this.tokenExpires = new Date(Date.now() + expiresIn * 1000);
    }
  }

  // Video operations

  async getVideos(
    filter?: VideoFilter,
    pagination?: PaginationOptions
  ): Promise<PaginatedResponse<Video>> {
    const params = {
      ...filter,
      ...pagination,
    };

    return this.request<PaginatedResponse<Video>>(
      HttpMethod.GET,
      'videos',
      params
    );
  }

  async getVideo(videoId: string): Promise<Video> {
    return this.request<Video>(HttpMethod.GET, `videos/${videoId}`);
  }

  async createVideo(video: Partial<Video>): Promise<Video> {
    return this.request<Video>(HttpMethod.POST, 'videos', undefined, video);
  }

  async updateVideo(videoId: string, updates: Partial<Video>): Promise<Video> {
    return this.request<Video>(
      HttpMethod.PATCH,
      `videos/${videoId}`,
      undefined,
      updates
    );
  }

  async deleteVideo(videoId: string): Promise<void> {
    await this.request<void>(HttpMethod.DELETE, `videos/${videoId}`);
  }

  // Channel operations

  async getChannels(): Promise<Channel[]> {
    return this.request<Channel[]>(HttpMethod.GET, 'channels');
  }

  async getChannel(channelId: string): Promise<Channel> {
    return this.request<Channel>(HttpMethod.GET, `channels/${channelId}`);
  }

  // Summary operations

  async getSummary(videoId: string): Promise<Summary> {
    return this.request<Summary>(HttpMethod.GET, `videos/${videoId}/summary`);
  }

  async createSummary(summary: Partial<Summary>): Promise<Summary> {
    return this.request<Summary>(
      HttpMethod.POST,
      'summaries',
      undefined,
      summary
    );
  }

  // Recommendation operations

  async getRecommendations(
    userId?: string,
    limit: number = 10
  ): Promise<Recommendation[]> {
    const params = { userId, limit };
    return this.request<Recommendation[]>(
      HttpMethod.GET,
      'recommendations',
      params
    );
  }

  // Analytics

  async getAnalytics(): Promise<Analytics> {
    return this.request<Analytics>(HttpMethod.GET, 'analytics');
  }

  // GraphQL support

  async graphql<T = any>(
    query: string,
    variables?: Record<string, any>
  ): Promise<T> {
    return this.request<T>(
      HttpMethod.POST,
      'graphql',
      undefined,
      { query, variables }
    );
  }

  // WebSocket support

  connect(): void {
    const wsUrl = this.config.baseUrl.replace(/^http/, 'ws') + '/ws';

    this.ws = new WebSocket(wsUrl, {
      headers: {
        'Authorization': this.accessToken ? `Bearer ${this.accessToken}` : '',
        'X-API-Key': this.config.apiKey,
      },
    });

    this.ws.on('open', () => {
      this.emit('connected');
      if (this.config.debug) {
        console.log('WebSocket connected');
      }
    });

    this.ws.on('message', (data: WebSocket.Data) => {
      try {
        const message = JSON.parse(data.toString());
        this.emit('message', message);

        if (message.type) {
          this.emit(message.type, message.data);
        }
      } catch (error) {
        this.emit('error', error);
      }
    });

    this.ws.on('error', (error) => {
      this.emit('error', error);
    });

    this.ws.on('close', () => {
      this.emit('disconnected');
      if (this.config.debug) {
        console.log('WebSocket disconnected');
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = undefined;
    }
  }

  subscribe(event: string, data?: any): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    this.ws.send(JSON.stringify({
      type: 'subscribe',
      event,
      data,
    }));
  }

  unsubscribe(event: string): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    this.ws.send(JSON.stringify({
      type: 'unsubscribe',
      event,
    }));
  }

  // Webhook signature verification

  static verifyWebhookSignature(
    payload: string | Buffer,
    signature: string,
    secret: string
  ): boolean {
    const expected = crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex');

    return crypto.timingSafeEqual(
      Buffer.from(expected),
      Buffer.from(signature)
    );
  }

  // Batch operations

  async batch(operations: BatchOperation[]): Promise<any[]> {
    const response = await this.request<{ results: any[] }>(
      HttpMethod.POST,
      'batch',
      undefined,
      { operations }
    );

    return response.results;
  }

  // Utility methods

  async healthCheck(): Promise<boolean> {
    try {
      await this.request<{ status: string }>(HttpMethod.GET, 'health');
      return true;
    } catch {
      return false;
    }
  }

  async getApiVersions(): Promise<any> {
    return this.request<any>(HttpMethod.GET, 'versions');
  }
}

// Helper functions

export function createClient(config?: ClientConfig): VideoAggregatorClient {
  return new VideoAggregatorClient(config);
}

export async function withClient<T>(
  config: ClientConfig,
  fn: (client: VideoAggregatorClient) => Promise<T>
): Promise<T> {
  const client = new VideoAggregatorClient(config);
  try {
    return await fn(client);
  } finally {
    client.disconnect();
  }
}

// Example usage

async function example() {
  const client = new VideoAggregatorClient({
    apiKey: 'your-api-key',
    debug: true,
  });

  try {
    // Get videos
    const videos = await client.getVideos(
      { minQualityScore: 0.7 },
      { limit: 10 }
    );

    console.log(`Found ${videos.totalCount} videos`);
    videos.data.forEach(video => {
      console.log(`- ${video.title} (Score: ${video.qualityScore})`);
    });

    // GraphQL query
    const result = await client.graphql(`
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
    `);

    // WebSocket subscription
    client.on('video.created', (video) => {
      console.log('New video:', video);
    });

    client.connect();
    client.subscribe('video.created');

    // Batch operations
    const batchResults = await client.batch([
      {
        method: HttpMethod.GET,
        endpoint: 'videos/1',
      },
      {
        method: HttpMethod.GET,
        endpoint: 'channels/1',
      },
    ]);

  } catch (error) {
    console.error('Error:', error);
  } finally {
    client.disconnect();
  }
}

// Export everything
export default VideoAggregatorClient;