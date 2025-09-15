import { useState } from 'react';
import { Play, Eye, Heart, MessageCircle, Clock, ExternalLink, FileText, Loader2 } from 'lucide-react';
import { formatNumber, formatRelativeTime, getPlatformColor, getPlatformName } from '@/lib/api';
import type { Video } from '@/types';

interface VideoCardProps {
  video: Video;
}

export function VideoCard({ video }: VideoCardProps) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  const handleImageLoad = () => {
    setImageLoaded(true);
  };

  const handleImageError = () => {
    setImageError(true);
    setImageLoaded(true);
  };

  const openVideo = () => {
    window.open(video.url, '_blank', 'noopener,noreferrer');
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border hover:shadow-md transition-shadow duration-200">
      {/* Thumbnail */}
      <div className="relative aspect-video bg-gray-100 rounded-t-lg overflow-hidden">
        {!imageLoaded && !imageError && (
          <div className="absolute inset-0 flex items-center justify-center">
            <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
          </div>
        )}

        {video.thumbnail_url && !imageError ? (
          <img
            src={video.thumbnail_url}
            alt={video.title}
            className={`w-full h-full object-cover transition-opacity duration-200 ${
              imageLoaded ? 'opacity-100' : 'opacity-0'
            }`}
            onLoad={handleImageLoad}
            onError={handleImageError}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gray-200">
            <Play className="h-12 w-12 text-gray-400" />
          </div>
        )}

        {/* Play button overlay */}
        <div
          className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-30 transition-opacity duration-200 cursor-pointer flex items-center justify-center group"
          onClick={openVideo}
        >
          <div className="bg-white bg-opacity-90 rounded-full p-3 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <Play className="h-6 w-6 text-gray-800" />
          </div>
        </div>

        {/* Duration badge */}
        {video.duration && (
          <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white text-xs px-2 py-1 rounded">
            {Math.floor(video.duration / 60)}:{(video.duration % 60).toString().padStart(2, '0')}
          </div>
        )}

        {/* Platform badge */}
        <div className="absolute top-2 left-2">
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPlatformColor(video.platform)}`}>
            {getPlatformName(video.platform)}
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Title */}
        <h3 className="font-semibold text-gray-900 line-clamp-2 mb-2 leading-tight">
          {video.title}
        </h3>

        {/* Author */}
        <p className="text-sm text-gray-600 mb-3">{video.author}</p>

        {/* Description */}
        {video.description && (
          <p className="text-sm text-gray-600 line-clamp-2 mb-3">
            {video.description}
          </p>
        )}

        {/* Stats */}
        <div className="flex items-center justify-between text-sm text-gray-500 mb-3">
          <div className="flex items-center space-x-4">
            <span className="flex items-center space-x-1">
              <Eye className="h-4 w-4" />
              <span>{formatNumber(video.view_count)}</span>
            </span>
            <span className="flex items-center space-x-1">
              <Heart className="h-4 w-4" />
              <span>{formatNumber(video.like_count)}</span>
            </span>
            <span className="flex items-center space-x-1">
              <MessageCircle className="h-4 w-4" />
              <span>{formatNumber(video.comment_count)}</span>
            </span>
          </div>
        </div>

        {/* Upload date */}
        <div className="flex items-center text-xs text-gray-400 mb-3">
          <Clock className="h-3 w-3 mr-1" />
          <span>{formatRelativeTime(video.upload_date)}</span>
        </div>

        {/* Transcription status */}
        {video.transcription && (
          <div className="mb-3">
            <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
              video.transcription.status === 'success'
                ? 'bg-green-100 text-green-800'
                : video.transcription.status === 'pending'
                ? 'bg-yellow-100 text-yellow-800'
                : video.transcription.status === 'failed'
                ? 'bg-red-100 text-red-800'
                : 'bg-gray-100 text-gray-800'
            }`}>
              <FileText className="h-3 w-3 mr-1" />
              <span>
                {video.transcription.status === 'success' ? 'Transcribed' :
                 video.transcription.status === 'pending' ? 'Processing' :
                 video.transcription.status === 'failed' ? 'Failed' : 'No transcript'}
              </span>
            </div>
          </div>
        )}

        {/* Tags */}
        {video.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-3">
            {video.tags.slice(0, 3).map((tag, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full"
              >
                {tag}
              </span>
            ))}
            {video.tags.length > 3 && (
              <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                +{video.tags.length - 3}
              </span>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-between pt-2 border-t">
          <button
            onClick={openVideo}
            className="flex items-center space-x-1 text-blue-600 hover:text-blue-700 text-sm font-medium"
          >
            <ExternalLink className="h-4 w-4" />
            <span>Watch</span>
          </button>

          {video.transcription?.status === 'success' && video.transcription.paragraph_summary && (
            <button className="text-gray-600 hover:text-gray-700 text-sm">
              <FileText className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* Transcription summary modal would go here if needed */}
    </div>
  );
}