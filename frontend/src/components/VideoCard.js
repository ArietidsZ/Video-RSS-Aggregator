import React, { useState } from 'react';
import {
  PlayIcon,
  EyeIcon,
  HeartIcon,
  ChatBubbleLeftIcon,
  ClockIcon,
  UserIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import { HeartIcon as HeartSolidIcon } from '@heroicons/react/24/solid';

function VideoCard({ video }) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  const formatNumber = (num) => {
    if (!num) return '0';
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  const formatDuration = (seconds) => {
    if (!seconds) return '';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatDate = (dateString) => {
    if (!dateString) return '';
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString();
    } catch {
      return '';
    }
  };

  const getPlatformColor = (platform) => {
    const colors = {
      bilibili: 'platform-bilibili',
      douyin: 'platform-douyin',
      kuaishou: 'platform-kuaishou',
    };
    return colors[platform] || 'bg-gray-100 text-gray-800';
  };

  const getSentimentColor = (sentiment) => {
    const colors = {
      positive: 'text-green-600',
      negative: 'text-red-600',
      neutral: 'text-gray-600',
    };
    return colors[sentiment] || 'text-gray-600';
  };

  const handleImageLoad = () => {
    setImageLoaded(true);
  };

  const handleImageError = () => {
    setImageError(true);
    setImageLoaded(true);
  };

  const openVideo = () => {
    if (video.url) {
      window.open(video.url, '_blank', 'noopener,noreferrer');
    }
  };

  return (
    <div className="video-card group cursor-pointer" onClick={openVideo}>
      {/* Thumbnail */}
      <div className="relative aspect-video bg-gray-200 rounded-lg overflow-hidden mb-3">
        {!imageLoaded && !imageError && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="loading-spinner"></div>
          </div>
        )}

        {video.thumbnail_url && !imageError ? (
          <img
            src={video.thumbnail_url}
            alt={video.title}
            className={`w-full h-full object-cover transition-all duration-300 group-hover:scale-105 ${
              imageLoaded ? 'opacity-100' : 'opacity-0'
            }`}
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-br from-gray-200 to-gray-300 flex items-center justify-center">
            <PlayIcon className="w-8 h-8 text-gray-400" />
          </div>
        )}

        {/* Play overlay */}
        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all duration-300 flex items-center justify-center">
          <PlayIcon className="w-12 h-12 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        </div>

        {/* Duration badge */}
        {video.duration && (
          <div className="absolute bottom-2 right-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
            {formatDuration(video.duration)}
          </div>
        )}

        {/* Platform badge */}
        <div className="absolute top-2 left-2">
          <span className={`platform-badge ${getPlatformColor(video.platform)}`}>
            {video.platform}
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-2">
        {/* Title */}
        <h3 className="font-medium text-gray-900 line-clamp-2 group-hover:text-primary-600 transition-colors duration-200">
          {video.title || 'Untitled Video'}
        </h3>

        {/* Author */}
        <div className="flex items-center text-sm text-gray-600">
          <UserIcon className="w-4 h-4 mr-1" />
          <span className="truncate">{video.author || 'Unknown Author'}</span>
        </div>

        {/* Stats */}
        <div className="flex items-center justify-between text-sm text-gray-500">
          <div className="flex items-center space-x-3">
            {video.views && (
              <div className="flex items-center">
                <EyeIcon className="w-4 h-4 mr-1" />
                <span>{formatNumber(video.views)}</span>
              </div>
            )}
            {video.likes && (
              <div className="flex items-center">
                <HeartIcon className="w-4 h-4 mr-1" />
                <span>{formatNumber(video.likes)}</span>
              </div>
            )}
            {video.comments && (
              <div className="flex items-center">
                <ChatBubbleLeftIcon className="w-4 h-4 mr-1" />
                <span>{formatNumber(video.comments)}</span>
              </div>
            )}
          </div>
          <div className="flex items-center">
            <ClockIcon className="w-4 h-4 mr-1" />
            <span>{formatDate(video.published_at)}</span>
          </div>
        </div>

        {/* AI Summary */}
        {video.summary && (
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-3 rounded-lg border border-blue-100">
            <div className="flex items-start space-x-2">
              <SparklesIcon className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
              <div className="space-y-1 flex-1">
                <p className="text-sm text-gray-700 line-clamp-3">
                  {video.summary}
                </p>
                {video.sentiment && (
                  <div className="flex items-center justify-between">
                    <span className={`text-xs font-medium ${getSentimentColor(video.sentiment)}`}>
                      {video.sentiment}
                    </span>
                    {video.confidence && (
                      <span className="text-xs text-gray-500">
                        {Math.round(video.confidence * 100)}% confidence
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Tags */}
        {video.tags && video.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {video.tags.slice(0, 3).map((tag, index) => (
              <span
                key={index}
                className="inline-block bg-gray-100 text-gray-600 text-xs px-2 py-1 rounded-full"
              >
                {tag}
              </span>
            ))}
            {video.tags.length > 3 && (
              <span className="inline-block text-gray-400 text-xs px-2 py-1">
                +{video.tags.length - 3} more
              </span>
            )}
          </div>
        )}

        {/* Key Points */}
        {video.key_points && video.key_points.length > 0 && (
          <div className="space-y-1">
            <h4 className="text-xs font-medium text-gray-700">Key Points:</h4>
            <ul className="text-xs text-gray-600 space-y-0.5">
              {video.key_points.slice(0, 2).map((point, index) => (
                <li key={index} className="flex items-start">
                  <span className="text-primary-500 mr-1">•</span>
                  <span className="line-clamp-1">{point}</span>
                </li>
              ))}
              {video.key_points.length > 2 && (
                <li className="text-gray-400">
                  +{video.key_points.length - 2} more points
                </li>
              )}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default VideoCard;