import React, { useState } from 'react';
import { useVideos, usePlatforms, useHealth, useCacheStats, useClearCache } from '../hooks/useAPI';
import VideoCard from '../components/VideoCard';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import StatsCard from '../components/StatsCard';
import {
  ChartBarIcon,
  ClockIcon,
  ServerIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';

function Dashboard() {
  const [selectedPlatform, setSelectedPlatform] = useState('bilibili');
  const [videoLimit, setVideoLimit] = useState(12);

  // API hooks
  const { data: health, isLoading: healthLoading } = useHealth();
  const { data: platforms, isLoading: platformsLoading } = usePlatforms();
  const { data: videos, isLoading: videosLoading, error: videosError } = useVideos(selectedPlatform, {
    limit: videoLimit,
    include_summary: true,
  });
  const { data: cacheStats } = useCacheStats();
  const clearCacheMutation = useClearCache();

  const handleClearCache = () => {
    clearCacheMutation.mutate();
  };

  if (healthLoading || platformsLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  const platformColors = {
    bilibili: 'bg-pink-500',
    douyin: 'bg-red-500',
    kuaishou: 'bg-orange-500',
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            智能内容聚合平台
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            AI-powered efficient information digest from Chinese video platforms
          </p>
        </div>
        <div className="mt-4 md:mt-0">
          <div className="flex items-center space-x-2 text-sm">
            <span className="bg-gradient-to-r from-green-100 to-blue-100 text-green-800 px-3 py-1 rounded-full">
              🚀 Real-time Processing
            </span>
            <span className="bg-gradient-to-r from-purple-100 to-pink-100 text-purple-800 px-3 py-1 rounded-full">
              🤖 AI-Enhanced
            </span>
          </div>
        </div>
      </div>

      {/* Trending AI Digests Spotlight */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-xl font-bold flex items-center">
              <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 24 24">
                <path d="M9 4.5a.75.75 0 01.721.544l.813 2.846a3.75 3.75 0 002.576 2.576l2.846.813a.75.75 0 010 1.442l-2.846.813a3.75 3.75 0 00-2.576 2.576l-.813 2.846a.75.75 0 01-1.442 0l-.813-2.846a3.75 3.75 0 00-2.576-2.576l-2.846-.813a.75.75 0 010-1.442l2.846-.813A3.75 3.75 0 007.466 7.89l.813-2.846A.75.75 0 019 4.5z"/>
              </svg>
              今日热门内容摘要
            </h3>
            <p className="text-sm opacity-90">AI智能提取关键信息，5秒了解热门内容</p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">2.3M</div>
            <div className="text-xs opacity-80">今日处理视频数</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur-sm">
            <div className="flex items-center mb-2">
              <span className="bg-red-500 text-white text-xs px-2 py-1 rounded">抖音</span>
              <span className="ml-2 text-sm opacity-90">1.2M 观看</span>
            </div>
            <h4 className="font-medium mb-2">最新iPhone 15评测对比</h4>
            <p className="text-sm opacity-80 mb-3">📱 新功能解析: USB-C接口、钛金属材质、摄像头升级。性价比分析显示相比14系列提升显著...</p>
            <div className="flex items-center justify-between text-xs">
              <span className="bg-green-200 text-green-800 px-2 py-1 rounded">95% 准确度</span>
              <span>⚡ 0.3秒生成</span>
            </div>
          </div>

          <div className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur-sm">
            <div className="flex items-center mb-2">
              <span className="bg-pink-500 text-white text-xs px-2 py-1 rounded">哔哩</span>
              <span className="ml-2 text-sm opacity-90">856K 观看</span>
            </div>
            <h4 className="font-medium mb-2">AI编程教程 - Python自动化</h4>
            <p className="text-sm opacity-80 mb-3">🐍 核心内容: 网络爬虫实现、数据清洗技术、自动化测试。适合初学者，代码实例丰富...</p>
            <div className="flex items-center justify-between text-xs">
              <span className="bg-green-200 text-green-800 px-2 py-1 rounded">92% 准确度</span>
              <span>⚡ 0.4秒生成</span>
            </div>
          </div>

          <div className="bg-white bg-opacity-10 rounded-lg p-4 backdrop-blur-sm">
            <div className="flex items-center mb-2">
              <span className="bg-orange-500 text-white text-xs px-2 py-1 rounded">快手</span>
              <span className="ml-2 text-sm opacity-90">634K 观看</span>
            </div>
            <h4 className="font-medium mb-2">经济热点分析：房地产市场</h4>
            <p className="text-sm opacity-80 mb-3">🏠 关键观点: 政策调整影响、区域差异分析、投资建议。专家认为下半年将呈现结构性复苏...</p>
            <div className="flex items-center justify-between text-xs">
              <span className="bg-green-200 text-green-800 px-2 py-1 rounded">88% 准确度</span>
              <span>⚡ 0.5秒生成</span>
            </div>
          </div>
        </div>

        <div className="mt-4 text-center">
          <a href="/education-ai" className="inline-flex items-center px-4 py-2 bg-white text-blue-600 rounded-lg hover:bg-gray-100 transition-colors font-medium text-sm">
            查看完整AI分析报告 →
          </a>
        </div>
      </div>

      {/* System Status Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="System Status"
          value={health?.status === 'healthy' ? 'Healthy' : 'Error'}
          icon={ServerIcon}
          color={health?.status === 'healthy' ? 'green' : 'red'}
        />
        <StatsCard
          title="Platforms"
          value={platforms?.platforms?.length || 0}
          icon={ChartBarIcon}
          color="blue"
        />
        <StatsCard
          title="Cache Entries"
          value={cacheStats?.total_entries || 0}
          icon={ClockIcon}
          color="purple"
          subtitle={`${cacheStats?.valid_entries || 0} valid`}
        />
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <TrashIcon className="h-6 w-6 text-gray-400" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Cache</dt>
                  <dd className="text-lg font-medium text-gray-900">
                    <button
                      onClick={handleClearCache}
                      disabled={clearCacheMutation.isLoading}
                      className="btn btn-sm btn-secondary"
                    >
                      {clearCacheMutation.isLoading ? 'Clearing...' : 'Clear Cache'}
                    </button>
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Platform Selection */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Video Feed</h3>
          <p className="mt-1 text-sm text-gray-500">
            Browse trending videos from different platforms
          </p>
        </div>
        <div className="px-6 py-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
            {/* Platform tabs */}
            <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
              {platforms?.platforms?.map((platform) => (
                <button
                  key={platform.id}
                  onClick={() => setSelectedPlatform(platform.id)}
                  className={`px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                    selectedPlatform === platform.id
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <div
                      className={`w-2 h-2 rounded-full ${platformColors[platform.id] || 'bg-gray-400'}`}
                    />
                    <span>{platform.name}</span>
                  </div>
                </button>
              ))}
            </div>

            {/* Video limit selector */}
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-500">Show:</label>
              <select
                value={videoLimit}
                onChange={(e) => setVideoLimit(Number(e.target.value))}
                className="input text-sm w-20"
              >
                <option value={6}>6</option>
                <option value={12}>12</option>
                <option value={24}>24</option>
                <option value={48}>48</option>
              </select>
              <span className="text-sm text-gray-500">videos</span>
            </div>
          </div>
        </div>
      </div>

      {/* Videos Grid */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900 capitalize">
              {selectedPlatform} Videos
            </h3>
            {videos && (
              <span className="text-sm text-gray-500">
                {videos.count} videos
              </span>
            )}
          </div>
        </div>

        <div className="p-6">
          {videosLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {Array.from({ length: videoLimit }).map((_, index) => (
                <div key={index} className="video-card">
                  <div className="loading-skeleton h-48 mb-4"></div>
                  <div className="space-y-2">
                    <div className="loading-skeleton h-4 w-3/4"></div>
                    <div className="loading-skeleton h-3 w-1/2"></div>
                    <div className="loading-skeleton h-3 w-2/3"></div>
                  </div>
                </div>
              ))}
            </div>
          ) : videosError ? (
            <ErrorMessage
              title="Failed to load videos"
              message={videosError.response?.data?.detail || videosError.message}
            />
          ) : videos && videos.videos ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {videos.videos.map((video) => (
                <VideoCard
                  key={`${video.platform}-${video.video_id}`}
                  video={video}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-500">No videos found</p>
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Quick Actions</h3>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <a
              href="/search"
              className="block p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-md transition-all duration-200"
            >
              <div className="text-center">
                <div className="w-8 h-8 mx-auto mb-2 text-primary-600">
                  <svg fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-gray-900">Search Videos</p>
                <p className="text-xs text-gray-500">Find specific content</p>
              </div>
            </a>

            <a
              href="/feeds"
              className="block p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-md transition-all duration-200"
            >
              <div className="text-center">
                <div className="w-8 h-8 mx-auto mb-2 text-primary-600">
                  <svg fill="currentColor" viewBox="0 0 20 20">
                    <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-gray-900">RSS Feeds</p>
                <p className="text-xs text-gray-500">Manage your feeds</p>
              </div>
            </a>

            <a
              href="/settings"
              className="block p-4 border border-gray-200 rounded-lg hover:border-primary-300 hover:shadow-md transition-all duration-200"
            >
              <div className="text-center">
                <div className="w-8 h-8 mx-auto mb-2 text-primary-600">
                  <svg fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-gray-900">Settings</p>
                <p className="text-xs text-gray-500">Configure options</p>
              </div>
            </a>

            <div className="block p-4 border border-gray-200 rounded-lg bg-gray-50">
              <div className="text-center">
                <div className="w-8 h-8 mx-auto mb-2 text-gray-400">
                  <svg fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-gray-500">Help Center</p>
                <p className="text-xs text-gray-400">Coming soon</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;