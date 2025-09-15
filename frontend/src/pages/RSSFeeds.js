import React, { useState } from 'react';
import {
  RssIcon,
  DocumentDuplicateIcon,
  EyeIcon,
  CheckIcon,
  Cog6ToothIcon,
  QrCodeIcon,
} from '@heroicons/react/24/outline';
import { getRSSFeedURL, getCombinedRSSFeedURL, usePlatforms } from '../hooks/useAPI';
import toast from 'react-hot-toast';
import QRCode from 'qrcode';

function RSSFeeds() {
  const [copied, setCopied] = useState('');
  const [feedSettings, setFeedSettings] = useState({
    limit: 20,
    include_summary: true,
    trending: true,
    platforms: ['bilibili', 'douyin', 'kuaishou'],
  });
  const [showQR, setShowQR] = useState('');
  const [qrDataUrl, setQrDataUrl] = useState('');

  const { data: platforms } = usePlatforms();

  const copyToClipboard = async (text, label) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(label);
      toast.success(`${label} copied to clipboard`);
      setTimeout(() => setCopied(''), 2000);
    } catch (err) {
      toast.error('Failed to copy to clipboard');
    }
  };

  const generateQR = async (url, label) => {
    try {
      const qrDataUrl = await QRCode.toDataURL(url, {
        width: 256,
        margin: 2,
        color: {
          dark: '#000000',
          light: '#FFFFFF',
        },
      });
      setQrDataUrl(qrDataUrl);
      setShowQR(label);
    } catch (err) {
      toast.error('Failed to generate QR code');
    }
  };

  const updateFeedSetting = (key, value) => {
    setFeedSettings(prev => ({ ...prev, [key]: value }));
  };

  const platformDetails = {
    bilibili: {
      name: 'Bilibili',
      description: 'Chinese video sharing platform',
      color: 'bg-pink-500',
      icon: '📺',
    },
    douyin: {
      name: 'Douyin',
      description: 'Chinese short video platform (TikTok)',
      color: 'bg-red-500',
      icon: '🎵',
    },
    kuaishou: {
      name: 'Kuaishou',
      description: 'Chinese short video platform',
      color: 'bg-orange-500',
      icon: '⚡',
    },
  };

  const feedTypes = [
    {
      id: 'individual',
      name: 'Individual Platform Feeds',
      description: 'Separate RSS feeds for each platform',
      icon: RssIcon,
    },
    {
      id: 'combined',
      name: 'Combined Feed',
      description: 'All platforms in one RSS feed',
      icon: RssIcon,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            RSS Feeds
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Generate and manage RSS feeds for video content across platforms
          </p>
        </div>
      </div>

      {/* Feed Configuration */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center space-x-2">
            <Cog6ToothIcon className="w-5 h-5 text-gray-400" />
            <h3 className="text-lg font-medium text-gray-900">Feed Configuration</h3>
          </div>
          <p className="mt-1 text-sm text-gray-500">
            Customize your RSS feed settings
          </p>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Videos per Feed
              </label>
              <select
                value={feedSettings.limit}
                onChange={(e) => updateFeedSetting('limit', Number(e.target.value))}
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
                Content Type
              </label>
              <select
                value={feedSettings.trending ? 'trending' : 'latest'}
                onChange={(e) => updateFeedSetting('trending', e.target.value === 'trending')}
                className="input"
              >
                <option value="trending">Trending</option>
                <option value="latest">Latest</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                AI Features
              </label>
              <label className="flex items-center mt-2">
                <input
                  type="checkbox"
                  checked={feedSettings.include_summary}
                  onChange={(e) => updateFeedSetting('include_summary', e.target.checked)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="ml-2 text-sm text-gray-700">Include AI summaries</span>
              </label>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Platforms
              </label>
              <div className="space-y-1">
                {platforms?.platforms?.map((platform) => (
                  <label key={platform.id} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={feedSettings.platforms.includes(platform.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          updateFeedSetting('platforms', [...feedSettings.platforms, platform.id]);
                        } else {
                          updateFeedSetting('platforms', feedSettings.platforms.filter(p => p !== platform.id));
                        }
                      }}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm text-gray-700">{platform.name}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Individual Platform Feeds */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Individual Platform Feeds</h3>
          <p className="mt-1 text-sm text-gray-500">
            Separate RSS feeds for each platform
          </p>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {feedSettings.platforms.map((platform) => {
              const details = platformDetails[platform];
              const feedUrl = getRSSFeedURL(platform, {
                limit: feedSettings.limit,
                include_summary: feedSettings.include_summary,
                trending: feedSettings.trending,
              });

              return (
                <div key={platform} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`w-10 h-10 rounded-lg ${details.color} flex items-center justify-center text-white text-lg`}>
                      {details.icon}
                    </div>
                    <div>
                      <h4 className="text-lg font-medium text-gray-900">{details.name}</h4>
                      <p className="text-sm text-gray-500">{details.description}</p>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <label className="block text-xs font-medium text-gray-500 mb-1">RSS URL</label>
                      <div className="flex items-center space-x-2">
                        <input
                          type="text"
                          value={feedUrl}
                          readOnly
                          className="input text-xs bg-gray-50 flex-1"
                        />
                        <button
                          onClick={() => copyToClipboard(feedUrl, `${platform} RSS`)}
                          className="btn btn-sm btn-secondary"
                          title="Copy URL"
                        >
                          {copied === `${platform} RSS` ? (
                            <CheckIcon className="w-4 h-4" />
                          ) : (
                            <DocumentDuplicateIcon className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>

                    <div className="flex space-x-2">
                      <button
                        onClick={() => window.open(feedUrl, '_blank')}
                        className="btn btn-sm btn-secondary flex-1"
                      >
                        <EyeIcon className="w-4 h-4 mr-1" />
                        Preview
                      </button>
                      <button
                        onClick={() => generateQR(feedUrl, `${platform} QR`)}
                        className="btn btn-sm btn-secondary"
                        title="Generate QR Code"
                      >
                        <QrCodeIcon className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Combined Feed */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">Combined Feed</h3>
          <p className="mt-1 text-sm text-gray-500">
            All selected platforms in one RSS feed
          </p>
        </div>
        <div className="p-6">
          {feedSettings.platforms.length > 0 ? (
            (() => {
              const combinedUrl = getCombinedRSSFeedURL({
                limit_per_platform: Math.floor(feedSettings.limit / feedSettings.platforms.length),
                include_summary: feedSettings.include_summary,
                platforms: feedSettings.platforms,
              });

              return (
                <div className="max-w-2xl">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-12 h-12 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center text-white text-xl">
                      🌐
                    </div>
                    <div>
                      <h4 className="text-lg font-medium text-gray-900">All Platforms Combined</h4>
                      <p className="text-sm text-gray-500">
                        Includes {feedSettings.platforms.join(', ')} with ~{Math.floor(feedSettings.limit / feedSettings.platforms.length)} videos each
                      </p>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">RSS URL</label>
                      <div className="flex items-center space-x-2">
                        <input
                          type="text"
                          value={combinedUrl}
                          readOnly
                          className="input bg-gray-50 flex-1"
                        />
                        <button
                          onClick={() => copyToClipboard(combinedUrl, 'Combined RSS')}
                          className="btn btn-secondary"
                          title="Copy URL"
                        >
                          {copied === 'Combined RSS' ? (
                            <CheckIcon className="w-5 h-5" />
                          ) : (
                            <DocumentDuplicateIcon className="w-5 h-5" />
                          )}
                        </button>
                      </div>
                    </div>

                    <div className="flex space-x-3">
                      <button
                        onClick={() => window.open(combinedUrl, '_blank')}
                        className="btn btn-primary"
                      >
                        <EyeIcon className="w-5 h-5 mr-2" />
                        Preview Feed
                      </button>
                      <button
                        onClick={() => generateQR(combinedUrl, 'Combined QR')}
                        className="btn btn-secondary"
                        title="Generate QR Code"
                      >
                        <QrCodeIcon className="w-5 h-5 mr-2" />
                        QR Code
                      </button>
                    </div>
                  </div>
                </div>
              );
            })()
          ) : (
            <div className="text-center py-8">
              <RssIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No platforms selected</h3>
              <p className="mt-1 text-sm text-gray-500">
                Select at least one platform to generate a combined feed
              </p>
            </div>
          )}
        </div>
      </div>

      {/* How to Use */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-medium text-gray-900">How to Use RSS Feeds</h3>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="text-md font-medium text-gray-900 mb-3">Popular RSS Readers</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>• <strong>Feedly</strong> - Web-based RSS reader</li>
                <li>• <strong>Inoreader</strong> - Advanced RSS reader</li>
                <li>• <strong>NewsBlur</strong> - Social RSS reader</li>
                <li>• <strong>Apple News</strong> - iOS native reader</li>
                <li>• <strong>Thunderbird</strong> - Desktop email client</li>
              </ul>
            </div>
            <div>
              <h4 className="text-md font-medium text-gray-900 mb-3">Feed Features</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li>• <strong>AI Summaries</strong> - Automatic content summarization</li>
                <li>• <strong>Video Stats</strong> - Views, likes, comments</li>
                <li>• <strong>Platform Tags</strong> - Source identification</li>
                <li>• <strong>Author Info</strong> - Creator details</li>
                <li>• <strong>Thumbnails</strong> - Video preview images</li>
              </ul>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="text-sm font-medium text-blue-900 mb-2">💡 Pro Tip</h4>
            <p className="text-sm text-blue-800">
              Use the QR codes to quickly add feeds to mobile RSS readers. Most apps support QR code scanning for easy feed addition.
            </p>
          </div>
        </div>
      </div>

      {/* QR Code Modal */}
      {showQR && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50">
          <div className="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
            <div className="mt-3 text-center">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                QR Code for {showQR}
              </h3>
              {qrDataUrl && (
                <img src={qrDataUrl} alt="QR Code" className="mx-auto mb-4" />
              )}
              <p className="text-sm text-gray-500 mb-4">
                Scan with your RSS reader app to add this feed
              </p>
              <button
                onClick={() => setShowQR('')}
                className="btn btn-primary"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default RSSFeeds;