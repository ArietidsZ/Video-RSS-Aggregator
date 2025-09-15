import React from 'react';
import {
  SparklesIcon,
  CpuChipIcon,
  GlobeAltIcon,
  ShieldCheckIcon,
  RocketLaunchIcon,
  HeartIcon,
} from '@heroicons/react/24/outline';

function About() {
  const features = [
    {
      icon: SparklesIcon,
      title: 'AI-Powered Summaries',
      description: 'Automatically generates concise summaries using OpenAI Whisper for transcription and GPT for summarization.',
    },
    {
      icon: GlobeAltIcon,
      title: 'Multi-Platform Support',
      description: 'Aggregates content from Bilibili, Douyin, and Kuaishou with platform-specific optimizations.',
    },
    {
      icon: CpuChipIcon,
      title: 'Smart Recommendations',
      description: 'Personalized content recommendations based on user behavior and content analysis.',
    },
    {
      icon: ShieldCheckIcon,
      title: 'Privacy Focused',
      description: 'Your data stays secure with optional cookie authentication and local processing capabilities.',
    },
    {
      icon: RocketLaunchIcon,
      title: 'High Performance',
      description: 'Built with async processing, Redis caching, and concurrent scraping for optimal speed.',
    },
  ];

  const techStack = [
    { category: 'Frontend', items: ['React 18', 'Tailwind CSS', 'React Query', 'React Router'] },
    { category: 'Backend', items: ['FastAPI', 'Python 3.11', 'AsyncIO', 'Pydantic'] },
    { category: 'AI/ML', items: ['OpenAI Whisper', 'OpenAI GPT', 'Transformers', 'LangChain'] },
    { category: 'Database', items: ['PostgreSQL', 'MongoDB', 'Redis', 'SQLAlchemy'] },
    { category: 'Infrastructure', items: ['Docker', 'Nginx', 'Prometheus', 'Grafana'] },
  ];

  const stats = [
    { label: 'Platforms Supported', value: '3+' },
    { label: 'Languages Supported', value: '10+' },
    { label: 'AI Models Integrated', value: '5+' },
    { label: 'Cache Hit Rate', value: '95%' },
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">
          Video RSS Aggregator
        </h1>
        <p className="mt-4 text-xl text-gray-600 max-w-2xl mx-auto">
          AI-powered RSS feeds for Chinese video platforms with automatic transcription,
          summarization, and intelligent content recommendations.
        </p>
        <div className="mt-6 flex justify-center">
          <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary-100 text-primary-800">
            <SparklesIcon className="w-4 h-4 mr-1" />
            Version 1.0.0
          </span>
        </div>
      </div>

      {/* Stats */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">System Statistics</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl font-bold text-primary-600">{stat.value}</div>
                <div className="text-sm text-gray-500">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">Key Features</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 bg-primary-100 rounded-lg flex items-center justify-center">
                  <feature.icon className="w-5 h-5 text-primary-600" />
                </div>
                <div>
                  <h3 className="text-sm font-medium text-gray-900">{feature.title}</h3>
                  <p className="mt-1 text-sm text-gray-500">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Technology Stack */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">Technology Stack</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {techStack.map((stack, index) => (
              <div key={index}>
                <h3 className="text-sm font-medium text-gray-900 mb-3">{stack.category}</h3>
                <div className="space-y-2">
                  {stack.items.map((item, itemIndex) => (
                    <span
                      key={itemIndex}
                      className="inline-block bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded mr-2 mb-1"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Architecture Overview */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">System Architecture</h2>
        </div>
        <div className="p-6">
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <GlobeAltIcon className="w-8 h-8 text-blue-600" />
                </div>
                <h3 className="text-sm font-medium text-gray-900">Data Collection</h3>
                <p className="text-xs text-gray-500 mt-1">
                  Concurrent scrapers for Bilibili, Douyin, and Kuaishou with rate limiting and retry logic.
                </p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <SparklesIcon className="w-8 h-8 text-purple-600" />
                </div>
                <h3 className="text-sm font-medium text-gray-900">AI Processing</h3>
                <p className="text-xs text-gray-500 mt-1">
                  Whisper transcription and GPT summarization with Chinese language optimization.
                </p>
              </div>
              <div className="text-center">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                  <RocketLaunchIcon className="w-8 h-8 text-green-600" />
                </div>
                <h3 className="text-sm font-medium text-gray-900">RSS Generation</h3>
                <p className="text-xs text-gray-500 mt-1">
                  Standards-compliant RSS 2.0 feeds with custom metadata and caching.
                </p>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-900 mb-2">Performance Optimizations</h4>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Async/await patterns for non-blocking I/O operations</li>
                <li>• Redis caching with 15-minute TTL for frequently accessed data</li>
                <li>• Concurrent video processing with semaphore-based rate limiting</li>
                <li>• Database connection pooling and query optimization</li>
                <li>• Nginx reverse proxy with gzip compression and static asset caching</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* Usage Guidelines */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">Usage Guidelines</h2>
        </div>
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-3">✅ Recommended Use</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Personal video content aggregation</li>
                <li>• Educational research and analysis</li>
                <li>• Content discovery and recommendations</li>
                <li>• AI-powered content summarization</li>
                <li>• RSS feed generation for personal use</li>
              </ul>
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-3">⚠️ Important Notes</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• Respect platform terms of service</li>
                <li>• Honor copyright and intellectual property</li>
                <li>• Use appropriate rate limiting</li>
                <li>• Educational and personal use only</li>
                <li>• Cookies are stored locally only</li>
              </ul>
            </div>
          </div>

          <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <h4 className="text-sm font-medium text-yellow-900 mb-2">⚖️ Legal Disclaimer</h4>
            <p className="text-sm text-yellow-800">
              This tool is for educational and personal use only. Users are responsible for complying
              with applicable laws and platform terms of service. The developers are not responsible
              for any misuse of this software.
            </p>
          </div>
        </div>
      </div>

      {/* API Documentation */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">API Documentation</h2>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            <p className="text-sm text-gray-600">
              The system provides a comprehensive REST API for programmatic access to all features.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-900 mb-2">RSS Endpoints</h4>
                <code className="text-xs text-gray-600 block">GET /rss/&#123;platform&#125;</code>
                <code className="text-xs text-gray-600 block">GET /rss/all</code>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-900 mb-2">Video API</h4>
                <code className="text-xs text-gray-600 block">GET /api/videos/&#123;platform&#125;</code>
                <code className="text-xs text-gray-600 block">POST /api/transcribe</code>
              </div>
            </div>

            <div className="flex space-x-4">
              <a
                href="/docs"
                target="_blank"
                className="btn btn-primary"
                rel="noopener noreferrer"
              >
                View API Docs
              </a>
              <a
                href="/redoc"
                target="_blank"
                className="btn btn-secondary"
                rel="noopener noreferrer"
              >
                ReDoc Documentation
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="text-center py-8">
        <div className="flex items-center justify-center space-x-2 text-gray-500">
          <span>Made with</span>
          <HeartIcon className="w-4 h-4 text-red-500" />
          <span>for the developer community</span>
        </div>
        <p className="mt-2 text-sm text-gray-400">
          Video RSS Aggregator v1.0.0 - Open source project
        </p>
      </div>
    </div>
  );
}

export default About;