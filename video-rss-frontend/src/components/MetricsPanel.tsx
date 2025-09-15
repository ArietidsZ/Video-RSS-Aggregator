import { BarChart3, Activity, Database, Cpu, HardDrive, Globe, Users, Clock } from 'lucide-react';
import { formatNumber } from '@/lib/api';
import type { ApiStats } from '@/types';

interface MetricsPanelProps {
  stats: ApiStats;
}

export function MetricsPanel({ stats }: MetricsPanelProps) {
  const metrics = [
    {
      title: 'Total Videos',
      value: formatNumber(stats.total_videos),
      icon: BarChart3,
      color: 'text-blue-600 bg-blue-100',
    },
    {
      title: 'Requests/Min',
      value: formatNumber(stats.requests_per_minute),
      icon: Activity,
      color: 'text-green-600 bg-green-100',
    },
    {
      title: 'Avg Response Time',
      value: `${stats.avg_response_time_ms}ms`,
      icon: Clock,
      color: 'text-yellow-600 bg-yellow-100',
    },
    {
      title: 'Cache Hit Rate',
      value: `${(stats.cache_hit_rate * 100).toFixed(1)}%`,
      icon: Database,
      color: 'text-purple-600 bg-purple-100',
    },
    {
      title: 'Queue Size',
      value: formatNumber(stats.transcription_queue_size),
      icon: Users,
      color: 'text-indigo-600 bg-indigo-100',
    },
  ];

  const systemMetrics = [
    {
      title: 'CPU Usage',
      value: `${stats.system_metrics.cpu_usage.toFixed(1)}%`,
      icon: Cpu,
      color: 'text-red-600 bg-red-100',
      progress: stats.system_metrics.cpu_usage,
    },
    {
      title: 'Memory Usage',
      value: `${stats.system_metrics.memory_usage.toFixed(1)}%`,
      icon: Activity,
      color: 'text-orange-600 bg-orange-100',
      progress: stats.system_metrics.memory_usage,
    },
    {
      title: 'Disk Usage',
      value: `${stats.system_metrics.disk_usage.toFixed(1)}%`,
      icon: HardDrive,
      color: 'text-teal-600 bg-teal-100',
      progress: stats.system_metrics.disk_usage,
    },
  ];

  const getProgressColor = (value: number) => {
    if (value < 50) return 'bg-green-500';
    if (value < 80) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-6">
      {/* Overview Metrics */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {metrics.map((metric, index) => {
            const Icon = metric.icon;
            return (
              <div key={index} className="bg-white p-6 rounded-lg shadow-sm border">
                <div className="flex items-center">
                  <div className={`p-2 rounded-full ${metric.color}`}>
                    <Icon className="h-6 w-6" />
                  </div>
                  <div className="ml-4">
                    <p className="text-2xl font-bold text-gray-900">{metric.value}</p>
                    <p className="text-sm text-gray-600">{metric.title}</p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* System Metrics */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">System Resources</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {systemMetrics.map((metric, index) => {
            const Icon = metric.icon;
            return (
              <div key={index} className="bg-white p-6 rounded-lg shadow-sm border">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <div className={`p-2 rounded-full ${metric.color}`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="ml-3">
                      <p className="text-lg font-semibold text-gray-900">{metric.value}</p>
                      <p className="text-sm text-gray-600">{metric.title}</p>
                    </div>
                  </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${getProgressColor(metric.progress)}`}
                    style={{ width: `${metric.progress}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Platform Distribution */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Platform Distribution</h2>
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="space-y-4">
            {Object.entries(stats.videos_by_platform).map(([platform, count]) => {
              const percentage = (count / stats.total_videos) * 100;
              return (
                <div key={platform}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700 capitalize">{platform}</span>
                    <span className="text-sm text-gray-600">{formatNumber(count)} ({percentage.toFixed(1)}%)</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="h-2 bg-blue-600 rounded-full transition-all duration-300"
                      style={{ width: `${percentage}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Platform Status */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Platform Status</h2>
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(stats.platform_status).map(([platform, isOnline]) => (
              <div key={platform} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center">
                  <Globe className="h-5 w-5 text-gray-400 mr-3" />
                  <span className="font-medium text-gray-700 capitalize">{platform}</span>
                </div>
                <div className={`flex items-center space-x-2 ${isOnline ? 'text-green-600' : 'text-red-600'}`}>
                  <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-600' : 'bg-red-600'}`} />
                  <span className="text-sm font-medium">{isOnline ? 'Online' : 'Offline'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Real-time Updates Notice */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center">
          <Activity className="h-5 w-5 text-blue-600 mr-2" />
          <span className="text-sm text-blue-800">
            Metrics are updated every 10 seconds automatically
          </span>
        </div>
      </div>
    </div>
  );
}