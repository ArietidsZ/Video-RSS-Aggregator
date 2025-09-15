import React, { useState, useEffect } from 'react';
import {
  CogIcon,
  KeyIcon,
  BellIcon,
  EyeIcon,
  ServerIcon,
  ShieldCheckIcon,
  DocumentDuplicateIcon,
  CheckIcon,
} from '@heroicons/react/24/outline';
import { useAutoDetectCookies, useUpdateCookies } from '../hooks/useAPI';
import toast from 'react-hot-toast';

function Settings() {
  const [activeTab, setActiveTab] = useState('general');
  const [settings, setSettings] = useState({
    // General settings
    defaultPlatform: 'bilibili',
    videoLimit: 20,
    includeAISummary: true,
    autoRefresh: false,
    refreshInterval: 15,

    // AI settings
    whisperModel: 'base',
    summaryMethod: 'auto',
    summaryLength: 150,
    language: 'zh',

    // Bilibili cookies
    bilibiliSessdata: '',
    bililiBiliJct: '',
    bilibiBuvid3: '',

    // RSS settings
    rssItemsPerFeed: 50,
    enableCache: true,
    cacheTTL: 15,

    // Notifications
    enableNotifications: false,
    notifyNewVideos: false,
    notifyErrors: true,
  });

  const [copied, setCopied] = useState('');
  const autoDetectMutation = useAutoDetectCookies();
  const updateCookiesMutation = useUpdateCookies();

  useEffect(() => {
    // Load settings from localStorage
    const savedSettings = localStorage.getItem('app_settings');
    if (savedSettings) {
      setSettings(prev => ({ ...prev, ...JSON.parse(savedSettings) }));
    }

    // Load cookies from localStorage
    const savedCookies = localStorage.getItem('bilibili_cookies');
    if (savedCookies) {
      const cookies = JSON.parse(savedCookies);
      setSettings(prev => ({
        ...prev,
        bilibiliSessdata: cookies.sessdata || '',
        bililiBiliJct: cookies.bili_jct || '',
        bilibiBuvid3: cookies.buvid3 || '',
      }));
    }
  }, []);

  const handleSettingChange = (key, value) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);

    // Save to localStorage
    localStorage.setItem('app_settings', JSON.stringify(newSettings));
  };

  const handleCookieDetection = async () => {
    try {
      await autoDetectMutation.mutateAsync();
      // Reload cookies from localStorage after detection
      const savedCookies = localStorage.getItem('bilibili_cookies');
      if (savedCookies) {
        const cookies = JSON.parse(savedCookies);
        setSettings(prev => ({
          ...prev,
          bilibiliSessdata: cookies.sessdata || '',
          bililiBiliJct: cookies.bili_jct || '',
          bilibiBuvid3: cookies.buvid3 || '',
        }));
      }
    } catch (error) {
      console.error('Cookie detection failed:', error);
    }
  };

  const handleCookieUpdate = () => {
    const cookies = {
      sessdata: settings.bilibiliSessdata,
      bili_jct: settings.bililiBiliJct,
      buvid3: settings.bilibiBuvid3,
    };
    updateCookiesMutation.mutate(cookies);
  };

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

  const tabs = [
    { id: 'general', name: 'General', icon: CogIcon },
    { id: 'cookies', name: 'Authentication', icon: KeyIcon },
    { id: 'ai', name: 'AI Settings', icon: ShieldCheckIcon },
    { id: 'rss', name: 'RSS Feeds', icon: ServerIcon },
    { id: 'notifications', name: 'Notifications', icon: BellIcon },
  ];

  const generateRSSUrl = (platform, options = {}) => {
    const baseUrl = window.location.origin;
    const params = new URLSearchParams({
      limit: options.limit || settings.videoLimit,
      include_summary: options.include_summary || settings.includeAISummary,
      trending: true,
    });
    return `${baseUrl}/rss/${platform}?${params}`;
  };

  const generateCombinedRSSUrl = () => {
    const baseUrl = window.location.origin;
    const params = new URLSearchParams({
      limit_per_platform: Math.floor(settings.rssItemsPerFeed / 3),
      include_summary: settings.includeAISummary,
    });
    return `${baseUrl}/rss/all?${params}`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Settings
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Configure your video RSS aggregator preferences
          </p>
        </div>
      </div>

      <div className="bg-white shadow rounded-lg">
        {/* Tab Navigation */}
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8 px-6" aria-label="Tabs">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`group inline-flex items-center py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-5 h-5 mr-2" />
                {tab.name}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {/* General Settings */}
          {activeTab === 'general' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">General Preferences</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Default Platform
                    </label>
                    <select
                      value={settings.defaultPlatform}
                      onChange={(e) => handleSettingChange('defaultPlatform', e.target.value)}
                      className="input"
                    >
                      <option value="bilibili">Bilibili</option>
                      <option value="douyin">Douyin</option>
                      <option value="kuaishou">Kuaishou</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Default Video Limit
                    </label>
                    <select
                      value={settings.videoLimit}
                      onChange={(e) => handleSettingChange('videoLimit', Number(e.target.value))}
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
                      Language
                    </label>
                    <select
                      value={settings.language}
                      onChange={(e) => handleSettingChange('language', e.target.value)}
                      className="input"
                    >
                      <option value="zh">中文 (Chinese)</option>
                      <option value="en">English</option>
                      <option value="auto">Auto-detect</option>
                    </select>
                  </div>
                  <div>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={settings.includeAISummary}
                        onChange={(e) => handleSettingChange('includeAISummary', e.target.checked)}
                        className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                      />
                      <span className="ml-2 text-sm text-gray-700">Include AI summaries by default</span>
                    </label>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-md font-medium text-gray-900 mb-3">Auto Refresh</h4>
                <div className="space-y-3">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.autoRefresh}
                      onChange={(e) => handleSettingChange('autoRefresh', e.target.checked)}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm text-gray-700">Enable auto refresh</span>
                  </label>
                  {settings.autoRefresh && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Refresh interval (minutes)
                      </label>
                      <select
                        value={settings.refreshInterval}
                        onChange={(e) => handleSettingChange('refreshInterval', Number(e.target.value))}
                        className="input w-32"
                      >
                        <option value={5}>5 minutes</option>
                        <option value={10}>10 minutes</option>
                        <option value={15}>15 minutes</option>
                        <option value={30}>30 minutes</option>
                        <option value={60}>1 hour</option>
                      </select>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Cookie Settings */}
          {activeTab === 'cookies' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">Bilibili Authentication</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Provide Bilibili cookies to access enhanced features and avoid rate limits.
                </p>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                  <h4 className="text-sm font-medium text-blue-900 mb-2">How to get cookies:</h4>
                  <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
                    <li>Open <a href="https://www.bilibili.com" target="_blank" rel="noopener noreferrer" className="underline">bilibili.com</a> and login</li>
                    <li>Open Developer Tools (F12)</li>
                    <li>Go to Application/Storage → Cookies → https://www.bilibili.com</li>
                    <li>Copy the values for SESSDATA, bili_jct, and buvid3</li>
                  </ol>
                </div>

                <div className="space-y-4">
                  <button
                    onClick={handleCookieDetection}
                    disabled={autoDetectMutation.isLoading}
                    className="btn btn-primary"
                  >
                    {autoDetectMutation.isLoading ? 'Detecting...' : 'Auto-detect Cookies'}
                  </button>

                  <div className="grid grid-cols-1 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        SESSDATA
                      </label>
                      <input
                        type="password"
                        value={settings.bilibiliSessdata}
                        onChange={(e) => handleSettingChange('bilibiliSessdata', e.target.value)}
                        placeholder="Enter SESSDATA cookie value"
                        className="input"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        bili_jct
                      </label>
                      <input
                        type="password"
                        value={settings.bililiBiliJct}
                        onChange={(e) => handleSettingChange('bililiBiliJct', e.target.value)}
                        placeholder="Enter bili_jct cookie value"
                        className="input"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        buvid3
                      </label>
                      <input
                        type="password"
                        value={settings.bilibiBuvid3}
                        onChange={(e) => handleSettingChange('bilibiBuvid3', e.target.value)}
                        placeholder="Enter buvid3 cookie value"
                        className="input"
                      />
                    </div>
                  </div>

                  <button
                    onClick={handleCookieUpdate}
                    disabled={updateCookiesMutation.isLoading}
                    className="btn btn-secondary"
                  >
                    {updateCookiesMutation.isLoading ? 'Updating...' : 'Update Cookies'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* AI Settings */}
          {activeTab === 'ai' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">AI Processing Settings</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Whisper Model
                    </label>
                    <select
                      value={settings.whisperModel}
                      onChange={(e) => handleSettingChange('whisperModel', e.target.value)}
                      className="input"
                    >
                      <option value="base">Base (faster)</option>
                      <option value="large-v3">Large-v3 (more accurate)</option>
                      <option value="turbo">Turbo (balanced)</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Larger models provide better accuracy but are slower
                    </p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Summary Method
                    </label>
                    <select
                      value={settings.summaryMethod}
                      onChange={(e) => handleSettingChange('summaryMethod', e.target.value)}
                      className="input"
                    >
                      <option value="auto">Auto (OpenAI then local)</option>
                      <option value="openai">OpenAI GPT</option>
                      <option value="local">Local model</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Summary Length (characters)
                    </label>
                    <input
                      type="number"
                      min="50"
                      max="500"
                      value={settings.summaryLength}
                      onChange={(e) => handleSettingChange('summaryLength', Number(e.target.value))}
                      className="input"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* RSS Settings */}
          {activeTab === 'rss' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">RSS Feed Configuration</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Items per Feed
                    </label>
                    <input
                      type="number"
                      min="10"
                      max="100"
                      value={settings.rssItemsPerFeed}
                      onChange={(e) => handleSettingChange('rssItemsPerFeed', Number(e.target.value))}
                      className="input"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Cache TTL (minutes)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="60"
                      value={settings.cacheTTL}
                      onChange={(e) => handleSettingChange('cacheTTL', Number(e.target.value))}
                      className="input"
                    />
                  </div>
                </div>
                <div className="mt-4">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.enableCache}
                      onChange={(e) => handleSettingChange('enableCache', e.target.checked)}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm text-gray-700">Enable caching</span>
                  </label>
                </div>
              </div>

              <div>
                <h4 className="text-md font-medium text-gray-900 mb-3">RSS Feed URLs</h4>
                <div className="space-y-3">
                  {['bilibili', 'douyin', 'kuaishou'].map((platform) => (
                    <div key={platform} className="flex items-center space-x-2">
                      <label className="block text-sm font-medium text-gray-700 w-20 capitalize">
                        {platform}:
                      </label>
                      <input
                        type="text"
                        value={generateRSSUrl(platform)}
                        readOnly
                        className="input flex-1 bg-gray-50"
                      />
                      <button
                        onClick={() => copyToClipboard(generateRSSUrl(platform), `${platform} RSS URL`)}
                        className="btn btn-sm btn-secondary"
                      >
                        {copied === `${platform} RSS URL` ? (
                          <CheckIcon className="w-4 h-4" />
                        ) : (
                          <DocumentDuplicateIcon className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                  ))}
                  <div className="flex items-center space-x-2">
                    <label className="block text-sm font-medium text-gray-700 w-20">
                      Combined:
                    </label>
                    <input
                      type="text"
                      value={generateCombinedRSSUrl()}
                      readOnly
                      className="input flex-1 bg-gray-50"
                    />
                    <button
                      onClick={() => copyToClipboard(generateCombinedRSSUrl(), 'Combined RSS URL')}
                      className="btn btn-sm btn-secondary"
                    >
                      {copied === 'Combined RSS URL' ? (
                        <CheckIcon className="w-4 h-4" />
                      ) : (
                        <DocumentDuplicateIcon className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Notifications */}
          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Notification Preferences</h3>
                <div className="space-y-4">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.enableNotifications}
                      onChange={(e) => handleSettingChange('enableNotifications', e.target.checked)}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm text-gray-700">Enable browser notifications</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.notifyNewVideos}
                      onChange={(e) => handleSettingChange('notifyNewVideos', e.target.checked)}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm text-gray-700">Notify when new videos are found</span>
                  </label>
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={settings.notifyErrors}
                      onChange={(e) => handleSettingChange('notifyErrors', e.target.checked)}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm text-gray-700">Notify on errors</span>
                  </label>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Settings;