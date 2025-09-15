import { useState } from 'react';
import { Save, RotateCcw, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';
import type { Config } from '@/types';

interface ConfigPanelProps {
  config: Config;
  onUpdate: (config: Partial<Config>) => void;
  isUpdating: boolean;
}

export function ConfigPanel({ config, onUpdate, isUpdating }: ConfigPanelProps) {
  const [localConfig, setLocalConfig] = useState<Config>(config);
  const [hasChanges, setHasChanges] = useState(false);

  const handleChange = (section: keyof Config, key: string, value: any) => {
    const newConfig = {
      ...localConfig,
      [section]: {
        ...localConfig[section],
        [key]: value,
      },
    };
    setLocalConfig(newConfig);
    setHasChanges(true);
  };

  const handlePlatformChange = (platform: string, key: string, value: any) => {
    const newConfig = {
      ...localConfig,
      platforms: {
        ...localConfig.platforms,
        [platform]: {
          ...localConfig.platforms[platform as keyof typeof localConfig.platforms],
          [key]: value,
        },
      },
    };
    setLocalConfig(newConfig);
    setHasChanges(true);
  };

  const handleSave = () => {
    onUpdate(localConfig);
    setHasChanges(false);
  };

  const handleReset = () => {
    setLocalConfig(config);
    setHasChanges(false);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold text-gray-900">Configuration</h2>
        <div className="flex space-x-3">
          <button
            onClick={handleReset}
            disabled={!hasChanges || isUpdating}
            className="flex items-center space-x-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 disabled:opacity-50"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Reset</span>
          </button>
          <button
            onClick={handleSave}
            disabled={!hasChanges || isUpdating}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {isUpdating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            <span>Save Changes</span>
          </button>
        </div>
      </div>

      {/* Changes Warning */}
      {hasChanges && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-yellow-600 mr-2" />
            <span className="text-sm text-yellow-800">
              You have unsaved changes. Click "Save Changes" to apply them.
            </span>
          </div>
        </div>
      )}

      {/* Platform Configuration */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Platform Settings</h3>
        <div className="space-y-6">
          {Object.entries(localConfig.platforms).map(([platform, settings]) => (
            <div key={platform} className="border rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-medium text-gray-900 capitalize">{platform}</h4>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={settings.enabled}
                    onChange={(e) => handlePlatformChange(platform, 'enabled', e.target.checked)}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Enabled</span>
                </label>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Rate Limit (per minute)
                  </label>
                  <input
                    type="number"
                    value={settings.rate_limit_per_minute}
                    onChange={(e) => handlePlatformChange(platform, 'rate_limit_per_minute', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Max Videos per Request
                  </label>
                  <input
                    type="number"
                    value={settings.max_videos_per_request}
                    onChange={(e) => handlePlatformChange(platform, 'max_videos_per_request', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Retry Attempts
                  </label>
                  <input
                    type="number"
                    value={settings.retry_attempts}
                    onChange={(e) => handlePlatformChange(platform, 'retry_attempts', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Timeout (seconds)
                  </label>
                  <input
                    type="number"
                    value={settings.timeout_seconds}
                    onChange={(e) => handlePlatformChange(platform, 'timeout_seconds', parseInt(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* RSS Configuration */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-medium text-gray-900 mb-4">RSS Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Feed Title
            </label>
            <input
              type="text"
              value={localConfig.rss.title}
              onChange={(e) => handleChange('rss', 'title', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Items
            </label>
            <input
              type="number"
              value={localConfig.rss.max_items}
              onChange={(e) => handleChange('rss', 'max_items', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="md:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description
            </label>
            <textarea
              value={localConfig.rss.description}
              onChange={(e) => handleChange('rss', 'description', e.target.value)}
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Cache TTL (seconds)
            </label>
            <input
              type="number"
              value={localConfig.rss.cache_ttl_seconds}
              onChange={(e) => handleChange('rss', 'cache_ttl_seconds', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={localConfig.rss.include_transcription}
                onChange={(e) => handleChange('rss', 'include_transcription', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-700">Include Transcriptions</span>
            </label>
          </div>
        </div>
      </div>

      {/* Transcription Configuration */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Transcription Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-center">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={localConfig.transcription.enabled}
                onChange={(e) => handleChange('transcription', 'enabled', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-700">Enable Transcription</span>
            </label>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model
            </label>
            <select
              value={localConfig.transcription.model}
              onChange={(e) => handleChange('transcription', 'model', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="whisper-large-v3">Whisper Large v3</option>
              <option value="sherpa-onnx-zh-en">Sherpa ONNX Chinese-English</option>
              <option value="faster-whisper-large-v2">Faster Whisper Large v2</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Language
            </label>
            <select
              value={localConfig.transcription.language}
              onChange={(e) => handleChange('transcription', 'language', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="auto">Auto-detect</option>
              <option value="zh">Chinese</option>
              <option value="en">English</option>
              <option value="zh-en">Chinese + English</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Max Concurrent Jobs
            </label>
            <input
              type="number"
              value={localConfig.transcription.max_concurrent}
              onChange={(e) => handleChange('transcription', 'max_concurrent', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Timeout (seconds)
            </label>
            <input
              type="number"
              value={localConfig.transcription.timeout_seconds}
              onChange={(e) => handleChange('transcription', 'timeout_seconds', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Cache Configuration */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Cache Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-center">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={localConfig.cache.redis_enabled}
                onChange={(e) => handleChange('cache', 'redis_enabled', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-700">Enable Redis</span>
            </label>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Redis URL
            </label>
            <input
              type="text"
              value={localConfig.cache.redis_url}
              onChange={(e) => handleChange('cache', 'redis_url', e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="redis://localhost:6379"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Memory Cache Size (MB)
            </label>
            <input
              type="number"
              value={localConfig.cache.memory_cache_size}
              onChange={(e) => handleChange('cache', 'memory_cache_size', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Default TTL (seconds)
            </label>
            <input
              type="number"
              value={localConfig.cache.default_ttl_seconds}
              onChange={(e) => handleChange('cache', 'default_ttl_seconds', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Rate Limiting Configuration */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Rate Limiting</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Requests per Second
            </label>
            <input
              type="number"
              value={localConfig.rate_limiting.requests_per_second}
              onChange={(e) => handleChange('rate_limiting', 'requests_per_second', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Burst Capacity
            </label>
            <input
              type="number"
              value={localConfig.rate_limiting.burst_capacity}
              onChange={(e) => handleChange('rate_limiting', 'burst_capacity', parseInt(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex items-center">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={localConfig.rate_limiting.enable_per_ip_limiting}
                onChange={(e) => handleChange('rate_limiting', 'enable_per_ip_limiting', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-700">Per-IP Limiting</span>
            </label>
          </div>
        </div>
      </div>

      {/* Success Message */}
      {!hasChanges && !isUpdating && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center">
            <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
            <span className="text-sm text-green-800">
              Configuration is up to date
            </span>
          </div>
        </div>
      )}
    </div>
  );
}