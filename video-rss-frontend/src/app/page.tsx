'use client';

import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Play, Pause, Download, RefreshCw, Loader2, Volume2, FileText, BarChart3, Settings } from 'lucide-react';
import { VideoCard } from '@/components/VideoCard';
import { MetricsPanel } from '@/components/MetricsPanel';
import { ConfigPanel } from '@/components/ConfigPanel';
import { api } from '@/lib/api';
import type { Video, ApiStats, Config } from '@/types';

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'videos' | 'metrics' | 'config'>('videos');
  const [selectedPlatforms, setSelectedPlatforms] = useState<string[]>(['bilibili', 'douyin', 'kuaishou']);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'upload_date' | 'view_count' | 'like_count'>('upload_date');

  const queryClient = useQueryClient();

  // Fetch videos
  const { data: videos = [], isLoading: videosLoading, error: videosError } = useQuery({
    queryKey: ['videos', selectedPlatforms, searchQuery, sortBy],
    queryFn: () => api.getVideos({
      platforms: selectedPlatforms,
      search: searchQuery,
      sort_by: sortBy,
      limit: 50
    }),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  // Fetch API statistics
  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: api.getStats,
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Fetch configuration
  const { data: config } = useQuery({
    queryKey: ['config'],
    queryFn: api.getConfig,
  });

  // Generate RSS mutation
  const generateRssMutation = useMutation({
    mutationFn: (platforms: string[]) => api.generateRss(platforms),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos'] });
    },
  });

  // Update configuration mutation
  const updateConfigMutation = useMutation({
    mutationFn: (newConfig: Partial<Config>) => api.updateConfig(newConfig),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] });
    },
  });

  // Handle RSS generation
  const handleGenerateRss = async () => {
    try {
      const response = await generateRssMutation.mutateAsync(selectedPlatforms);
      const blob = new Blob([response], { type: 'application/rss+xml' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `video-rss-${selectedPlatforms.join('-')}.xml`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to generate RSS:', error);
    }
  };

  // Filter videos based on search query
  const filteredVideos = videos.filter(video =>
    video.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    video.author.toLowerCase().includes(searchQuery.toLowerCase()) ||
    video.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Play className="h-8 w-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">Video RSS</h1>
              </div>
              {stats && (
                <div className="hidden sm:flex items-center space-x-4 text-sm text-gray-600">
                  <span className="flex items-center space-x-1">
                    <Volume2 className="h-4 w-4" />
                    <span>{stats.total_videos} videos</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <BarChart3 className="h-4 w-4" />
                    <span>{stats.requests_per_minute} req/min</span>
                  </span>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-4">
              <button
                onClick={handleGenerateRss}
                disabled={generateRssMutation.isPending}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                {generateRssMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Download className="h-4 w-4" />
                )}
                <span>Generate RSS</span>
              </button>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex space-x-8 border-b">
            {[
              { id: 'videos', label: 'Videos', icon: Play },
              { id: 'metrics', label: 'Metrics', icon: BarChart3 },
              { id: 'config', label: 'Settings', icon: Settings },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="h-5 w-5" />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'videos' && (
          <div className="space-y-6">
            {/* Filters */}
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {/* Search */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Search
                  </label>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search videos..."
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                {/* Platforms */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Platforms
                  </label>
                  <select
                    multiple
                    value={selectedPlatforms}
                    onChange={(e) => setSelectedPlatforms(Array.from(e.target.selectedOptions, option => option.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="bilibili">Bilibili</option>
                    <option value="douyin">Douyin</option>
                    <option value="kuaishou">Kuaishou</option>
                  </select>
                </div>

                {/* Sort */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Sort by
                  </label>
                  <select
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value as any)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="upload_date">Upload Date</option>
                    <option value="view_count">View Count</option>
                    <option value="like_count">Like Count</option>
                  </select>
                </div>

                {/* Refresh */}
                <div className="flex items-end">
                  <button
                    onClick={() => queryClient.invalidateQueries({ queryKey: ['videos'] })}
                    className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
                  >
                    <RefreshCw className="h-4 w-4" />
                    <span>Refresh</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Videos Grid */}
            {videosLoading ? (
              <div className="flex justify-center items-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
              </div>
            ) : videosError ? (
              <div className="text-center py-12">
                <p className="text-red-600">Failed to load videos</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredVideos.map((video) => (
                  <VideoCard key={video.id} video={video} />
                ))}
              </div>
            )}

            {filteredVideos.length === 0 && !videosLoading && (
              <div className="text-center py-12">
                <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">No videos found</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'metrics' && stats && (
          <MetricsPanel stats={stats} />
        )}

        {activeTab === 'config' && config && (
          <ConfigPanel
            config={config}
            onUpdate={(newConfig) => updateConfigMutation.mutate(newConfig)}
            isUpdating={updateConfigMutation.isPending}
          />
        )}
      </main>
    </div>
  );
}
