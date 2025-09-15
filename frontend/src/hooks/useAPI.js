import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import toast from 'react-hot-toast';

const api = axios.create({
  baseURL: process.env.NODE_ENV === 'production' ? '/api' : '/api',
  timeout: 30000,
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth headers here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      // Server responded with error status
      const message = error.response.data?.detail || error.response.data?.message || 'An error occurred';
      console.error('API Error:', message);
    } else if (error.request) {
      // Request was made but no response received
      console.error('Network Error:', error.message);
    } else {
      // Something else happened
      console.error('Error:', error.message);
    }
    return Promise.reject(error);
  }
);

// Health check
export const useHealth = () => {
  return useQuery({
    queryKey: ['health'],
    queryFn: async () => {
      const { data } = await api.get('/health');
      return data;
    },
    refetchInterval: 30000, // Check every 30 seconds
  });
};

// Get platforms
export const usePlatforms = () => {
  return useQuery({
    queryKey: ['platforms'],
    queryFn: async () => {
      const { data } = await api.get('/platforms');
      return data;
    },
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Get videos from a platform
export const useVideos = (platform, options = {}) => {
  const { limit = 10, include_summary = false, search = '' } = options;

  return useQuery({
    queryKey: ['videos', platform, { limit, include_summary, search }],
    queryFn: async () => {
      const params = new URLSearchParams({
        limit: limit.toString(),
        include_summary: include_summary.toString(),
      });

      if (search) {
        params.append('search', search);
      }

      const { data } = await api.get(`/videos/${platform}?${params}`);
      return data;
    },
    enabled: !!platform,
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

// Get RSS feed URL
export const getRSSFeedURL = (platform, options = {}) => {
  const { limit = 10, include_summary = false, trending = true } = options;
  const params = new URLSearchParams({
    limit: limit.toString(),
    include_summary: include_summary.toString(),
    trending: trending.toString(),
  });

  const baseURL = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';
  return `${baseURL}/rss/${platform}?${params}`;
};

// Get combined RSS feed URL
export const getCombinedRSSFeedURL = (options = {}) => {
  const { limit_per_platform = 5, include_summary = false, platforms = [] } = options;
  const params = new URLSearchParams({
    limit_per_platform: limit_per_platform.toString(),
    include_summary: include_summary.toString(),
  });

  if (platforms.length > 0) {
    params.append('platforms', platforms.join(','));
  }

  const baseURL = process.env.NODE_ENV === 'production' ? '' : 'http://localhost:8000';
  return `${baseURL}/rss/all?${params}`;
};

// Transcribe video
export const useTranscribeVideo = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ video_url, language = 'zh' }) => {
      const { data } = await api.post('/transcribe', {
        video_url,
        language,
      });
      return data;
    },
    onSuccess: () => {
      toast.success('Video transcribed successfully!');
    },
    onError: (error) => {
      const message = error.response?.data?.detail || 'Failed to transcribe video';
      toast.error(message);
    },
  });
};

// Summarize text
export const useSummarizeText = () => {
  return useMutation({
    mutationFn: async ({ text, method = 'auto', max_length = 150, min_length = 50, language = 'zh' }) => {
      const { data } = await api.post('/summarize', {
        text,
        method,
        max_length,
        min_length,
        language,
      });
      return data;
    },
    onSuccess: () => {
      toast.success('Text summarized successfully!');
    },
    onError: (error) => {
      const message = error.response?.data?.detail || 'Failed to summarize text';
      toast.error(message);
    },
  });
};

// Cache management
export const useCacheStats = () => {
  return useQuery({
    queryKey: ['cache-stats'],
    queryFn: async () => {
      const { data } = await api.get('/cache/stats');
      return data;
    },
    refetchInterval: 10000, // Update every 10 seconds
  });
};

export const useClearCache = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async () => {
      const { data } = await api.delete('/cache');
      return data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries(['cache-stats']);
      toast.success(data.message || 'Cache cleared successfully!');
    },
    onError: (error) => {
      const message = error.response?.data?.detail || 'Failed to clear cache';
      toast.error(message);
    },
  });
};

// Auto-detect cookies (custom implementation)
export const useAutoDetectCookies = () => {
  return useMutation({
    mutationFn: async () => {
      // This would typically involve a browser extension or manual input
      // For now, we'll simulate the process
      const cookies = await detectBilibiliCookies();
      return cookies;
    },
    onSuccess: (cookies) => {
      if (cookies.sessdata) {
        toast.success('Bilibili cookies detected successfully!');
        // Store cookies in localStorage or send to backend
        localStorage.setItem('bilibili_cookies', JSON.stringify(cookies));
      } else {
        toast.error('No valid Bilibili cookies found. Please login to Bilibili in another tab.');
      }
    },
    onError: () => {
      toast.error('Failed to detect cookies. Please check your Bilibili login status.');
    },
  });
};

// Helper function to detect Bilibili cookies
const detectBilibiliCookies = async () => {
  try {
    // In a real implementation, this would use a browser extension
    // or guide the user through manual cookie extraction

    // Check if cookies are already stored
    const stored = localStorage.getItem('bilibili_cookies');
    if (stored) {
      return JSON.parse(stored);
    }

    // For demo purposes, return empty cookies
    // In production, this would involve:
    // 1. Browser extension to read cookies
    // 2. Manual cookie input form
    // 3. Headless browser automation (server-side)

    return {
      sessdata: '',
      bili_jct: '',
      buvid3: '',
    };
  } catch (error) {
    console.error('Cookie detection error:', error);
    return {};
  }
};

// Update backend with new cookies
export const useUpdateCookies = () => {
  return useMutation({
    mutationFn: async (cookies) => {
      // This would send cookies to backend to update configuration
      // For now, we'll just store them locally
      localStorage.setItem('bilibili_cookies', JSON.stringify(cookies));
      return { success: true };
    },
    onSuccess: () => {
      toast.success('Cookies updated successfully!');
    },
    onError: () => {
      toast.error('Failed to update cookies');
    },
  });
};

export default api;