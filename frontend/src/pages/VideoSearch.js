import React, { useState, useEffect } from 'react';
import { MagnifyingGlassIcon, FunnelIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { useVideos, usePlatforms } from '../hooks/useAPI';
import VideoCard from '../components/VideoCard';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import { useDebounce } from '../hooks/useDebounce';

function VideoSearch() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedPlatform, setSelectedPlatform] = useState('bilibili');
  const [includeAI, setIncludeAI] = useState(true);
  const [videoLimit, setVideoLimit] = useState(20);
  const [showFilters, setShowFilters] = useState(false);

  // Debounce search query to avoid excessive API calls
  const debouncedSearchQuery = useDebounce(searchQuery, 500);

  const { data: platforms } = usePlatforms();

  // Search videos when query or platform changes
  const {
    data: searchResults,
    isLoading: searchLoading,
    error: searchError,
    refetch: refetchSearch
  } = useVideos(selectedPlatform, {
    search: debouncedSearchQuery,
    limit: videoLimit,
    include_summary: includeAI,
  });

  const handleSearch = (e) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      refetchSearch();
    }
  };

  const clearSearch = () => {
    setSearchQuery('');
  };

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
            Video Search
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Search for videos across Chinese platforms with AI-powered insights
          </p>
        </div>
        <div className="mt-4 md:mt-0">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="btn btn-secondary flex items-center space-x-2"
          >
            <FunnelIcon className="w-4 h-4" />
            <span>Filters</span>
          </button>
        </div>
      </div>

      {/* Search Form */}
      <div className="bg-white shadow rounded-lg p-6">
        <form onSubmit={handleSearch} className="space-y-4">
          {/* Search Input */}
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search for videos... (e.g., 游戏, 美食, 科技)"
              className="block w-full pl-10 pr-12 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-lg"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={clearSearch}
                className="absolute inset-y-0 right-0 pr-3 flex items-center"
              >
                <XMarkIcon className="h-5 w-5 text-gray-400 hover:text-gray-600" />
              </button>
            )}
          </div>

          {/* Platform Selection */}
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Platform:</label>
            <div className="flex space-x-2">
              {platforms?.platforms?.map((platform) => (
                <button
                  key={platform.id}
                  type="button"
                  onClick={() => setSelectedPlatform(platform.id)}
                  className={`px-4 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                    selectedPlatform === platform.id
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        selectedPlatform === platform.id
                          ? 'bg-white'
                          : platformColors[platform.id] || 'bg-gray-400'
                      }`}
                    />
                    <span>{platform.name}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Search Button */}
          <div className="flex justify-center">
            <button
              type="submit"
              disabled={!searchQuery.trim() || searchLoading}
              className="btn btn-primary btn-lg px-8"
            >
              {searchLoading ? (
                <div className="flex items-center space-x-2">
                  <LoadingSpinner size="sm" />
                  <span>Searching...</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <MagnifyingGlassIcon className="w-5 h-5" />
                  <span>Search Videos</span>
                </div>
              )}
            </button>
          </div>
        </form>

        {/* Filters Panel */}
        {showFilters && (
          <div className="mt-6 pt-6 border-t border-gray-200">
            <h3 className="text-sm font-medium text-gray-900 mb-4">Search Filters</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Results Limit
                </label>
                <select
                  value={videoLimit}
                  onChange={(e) => setVideoLimit(Number(e.target.value))}
                  className="input"
                >
                  <option value={10}>10 videos</option>
                  <option value={20}>20 videos</option>
                  <option value={30}>30 videos</option>
                  <option value={50}>50 videos</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  AI Features
                </label>
                <div className="flex items-center space-x-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={includeAI}
                      onChange={(e) => setIncludeAI(e.target.checked)}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm text-gray-700">Include AI summaries</span>
                  </label>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Quick Filters
                </label>
                <div className="space-y-1">
                  {['Gaming', 'Music', 'Food', 'Tech', 'Lifestyle'].map((tag) => (
                    <button
                      key={tag}
                      type="button"
                      onClick={() => setSearchQuery(tag)}
                      className="text-sm text-primary-600 hover:text-primary-800 block"
                    >
                      #{tag}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Search Results */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">
              {debouncedSearchQuery ? (
                <span>
                  Search Results for "{debouncedSearchQuery}" on{' '}
                  <span className="capitalize">{selectedPlatform}</span>
                </span>
              ) : (
                <span>Enter a search query to find videos</span>
              )}
            </h3>
            {searchResults && (
              <span className="text-sm text-gray-500">
                {searchResults.count} results
              </span>
            )}
          </div>
        </div>

        <div className="p-6">
          {!debouncedSearchQuery ? (
            // No search query
            <div className="text-center py-12">
              <MagnifyingGlassIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">Start searching</h3>
              <p className="mt-1 text-sm text-gray-500">
                Enter keywords to search for videos across platforms
              </p>
              <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Popular searches:</h4>
                <div className="flex flex-wrap justify-center gap-2">
                  {[
                    '游戏解说', '美食制作', '科技评测', '音乐翻唱', '日常vlog',
                    '学习教程', '旅游攻略', '健身运动', '搞笑视频', '新闻资讯'
                  ].map((term) => (
                    <button
                      key={term}
                      onClick={() => setSearchQuery(term)}
                      className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors"
                    >
                      {term}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : searchLoading ? (
            // Loading state
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {Array.from({ length: 8 }).map((_, index) => (
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
          ) : searchError ? (
            // Error state
            <ErrorMessage
              title="Search failed"
              message={searchError.response?.data?.detail || searchError.message}
              onRetry={() => refetchSearch()}
            />
          ) : searchResults && searchResults.videos && searchResults.videos.length > 0 ? (
            // Results found
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {searchResults.videos.map((video) => (
                  <VideoCard
                    key={`${video.platform}-${video.video_id}`}
                    video={video}
                  />
                ))}
              </div>

              {/* Search Stats */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-bold text-gray-900">
                      {searchResults.count}
                    </div>
                    <div className="text-sm text-gray-500">Total Results</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-gray-900">
                      {searchResults.videos.filter(v => v.summary).length}
                    </div>
                    <div className="text-sm text-gray-500">With AI Summary</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-gray-900">
                      {searchResults.videos.reduce((sum, v) => sum + (v.views || 0), 0).toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-500">Total Views</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-gray-900 capitalize">
                      {selectedPlatform}
                    </div>
                    <div className="text-sm text-gray-500">Platform</div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // No results
            <div className="text-center py-12">
              <MagnifyingGlassIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No results found</h3>
              <p className="mt-1 text-sm text-gray-500">
                Try adjusting your search terms or selecting a different platform
              </p>
              <div className="mt-6">
                <button
                  onClick={() => setSearchQuery('')}
                  className="btn btn-secondary"
                >
                  Clear search
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default VideoSearch;