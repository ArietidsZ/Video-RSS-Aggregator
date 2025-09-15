import React, { useState } from 'react';
import {
  AcademicCapIcon,
  ChartBarIcon,
  LightBulbIcon,
  UsersIcon,
  BookOpenIcon,
  StarIcon,
  GlobeAltIcon,
  SparklesIcon,
  ArrowTrendingUpIcon,
  HeartIcon
} from '@heroicons/react/24/outline';
import StatsCard from '../components/StatsCard';
import LoadingSpinner from '../components/LoadingSpinner';

function EducationAI() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [analyzeMode, setAnalyzeMode] = useState('overview');

  // Mock data for demo - in real app this would come from API
  const educationalMetrics = {
    totalVideos: 12847,
    educationalVideos: 3269,
    learners: 48392,
    subjects: 23,
    languages: 8,
    impactScore: 94.7
  };

  const subjectCategories = [
    { name: '数学 Math', count: 487, growth: 15.3, difficulty: 'Medium', color: 'bg-blue-500' },
    { name: '科学 Science', count: 623, growth: 22.1, difficulty: 'Hard', color: 'bg-green-500' },
    { name: '语言 Language', count: 392, growth: 8.7, difficulty: 'Easy', color: 'bg-purple-500' },
    { name: '历史 History', count: 234, growth: 12.4, difficulty: 'Medium', color: 'bg-yellow-500' },
    { name: '编程 Programming', count: 578, growth: 31.2, difficulty: 'Hard', color: 'bg-red-500' },
    { name: '艺术 Arts', count: 445, growth: 18.6, difficulty: 'Easy', color: 'bg-pink-500' }
  ];

  const aiRecommendations = [
    {
      title: "高考数学重难点解析",
      platform: "bilibili",
      views: "125万",
      aiScore: 95,
      difficulty: "高难度",
      topics: ["函数", "导数", "立体几何"],
      reason: "基于你的学习历史，这个视频能帮助突破数学难点"
    },
    {
      title: "Python编程入门教程",
      platform: "bilibili",
      views: "89万",
      aiScore: 91,
      difficulty: "初级",
      topics: ["基础语法", "数据结构", "算法"],
      reason: "适合编程初学者，循序渐进的教学方式"
    },
    {
      title: "英语口语练习 - 日常对话",
      platform: "douyin",
      views: "67万",
      aiScore: 88,
      difficulty: "中级",
      topics: ["口语", "听力", "交流"],
      reason: "实用性强，能快速提升英语交流能力"
    }
  ];

  const socialImpact = {
    accessibilityFeatures: [
      { feature: "自动字幕生成", coverage: "98%", users: 15429 },
      { feature: "多语言翻译", coverage: "12种语言", users: 8934 },
      { feature: "盲人语音描述", coverage: "45%", users: 2847 },
      { feature: "学习困难支持", coverage: "AI辅助", users: 6721 }
    ],
    regions: [
      { region: "偏远地区", users: 12847, improvement: "78%" },
      { region: "经济欠发达地区", users: 19234, improvement: "65%" },
      { region: "特殊教育群体", users: 4829, improvement: "89%" }
    ]
  };

  const difficultyColors = {
    'Easy': 'bg-green-100 text-green-800',
    'Medium': 'bg-yellow-100 text-yellow-800',
    'Hard': 'bg-red-100 text-red-800',
    '初级': 'bg-green-100 text-green-800',
    '中级': 'bg-yellow-100 text-yellow-800',
    '高难度': 'bg-red-100 text-red-800'
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg p-6 text-white">
        <div className="flex items-center space-x-3 mb-4">
          <SparklesIcon className="w-8 h-8" />
          <h1 className="text-3xl font-bold">AI 智能内容摘要平台</h1>
        </div>
        <p className="text-lg opacity-90">
          利用人工智能技术快速处理海量视频内容，提供高效信息摘要和智能推荐服务
        </p>
        <div className="mt-4 flex items-center space-x-6 text-sm">
          <div className="flex items-center space-x-1">
            <GlobeAltIcon className="w-4 h-4" />
            <span>服务全球 {educationalMetrics.learners.toLocaleString()} 名学习者</span>
          </div>
          <div className="flex items-center space-x-1">
            <HeartIcon className="w-4 h-4" />
            <span>社会影响力评分: {educationalMetrics.impactScore}/100</span>
          </div>
        </div>
      </div>

      {/* Mode Selection */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg mb-6">
          {[
            { id: 'overview', label: '概览 Overview', icon: ChartBarIcon },
            { id: 'recommendations', label: '智能推荐 AI Recommendations', icon: LightBulbIcon },
            { id: 'impact', label: '社会影响 Social Impact', icon: UsersIcon }
          ].map((mode) => (
            <button
              key={mode.id}
              onClick={() => setAnalyzeMode(mode.id)}
              className={`flex items-center space-x-2 px-4 py-2 text-sm font-medium rounded-md transition-all ${
                analyzeMode === mode.id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              <mode.icon className="w-4 h-4" />
              <span>{mode.label}</span>
            </button>
          ))}
        </div>

        {/* Overview Mode */}
        {analyzeMode === 'overview' && (
          <div className="space-y-6">
            {/* Educational Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatsCard
                title="教育视频总量"
                value={educationalMetrics.educationalVideos.toLocaleString()}
                subtitle={`占比 ${Math.round((educationalMetrics.educationalVideos / educationalMetrics.totalVideos) * 100)}%`}
                icon={BookOpenIcon}
                color="blue"
              />
              <StatsCard
                title="学科分类"
                value={educationalMetrics.subjects}
                subtitle="AI智能分类"
                icon={AcademicCapIcon}
                color="green"
              />
              <StatsCard
                title="受益学习者"
                value={educationalMetrics.learners.toLocaleString()}
                subtitle="月活跃用户"
                icon={UsersIcon}
                color="purple"
              />
              <StatsCard
                title="多语言支持"
                value={educationalMetrics.languages}
                subtitle="无障碍访问"
                icon={GlobeAltIcon}
                color="orange"
              />
            </div>

            {/* Subject Categories */}
            <div>
              <h3 className="text-lg font-semibold mb-4">学科内容分布与增长趋势</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {subjectCategories.map((subject, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${subject.color}`}></div>
                        <span className="font-medium">{subject.name}</span>
                      </div>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${difficultyColors[subject.difficulty]}`}>
                        {subject.difficulty}
                      </span>
                    </div>
                    <div className="text-2xl font-bold text-gray-900 mb-1">
                      {subject.count}
                    </div>
                    <div className="flex items-center text-sm text-green-600">
                      <ArrowTrendingUpIcon className="w-4 h-4 mr-1" />
                      <span>+{subject.growth}% 本月增长</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Analysis Features */}
            <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <SparklesIcon className="w-5 h-5 mr-2 text-purple-600" />
                AI 智能分析能力
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-600 mb-1">95%</div>
                  <div className="text-sm text-gray-600">内容分类准确率</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600 mb-1">87%</div>
                  <div className="text-sm text-gray-600">学习效果预测</div>
                </div>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600 mb-1">92%</div>
                  <div className="text-sm text-gray-600">个性化推荐匹配度</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Recommendations Mode */}
        {analyzeMode === 'recommendations' && (
          <div className="space-y-6">
            <div className="text-center">
              <h3 className="text-xl font-semibold mb-2">🎯 AI 个性化学习推荐</h3>
              <p className="text-gray-600">基于学习历史和AI分析的智能推荐</p>
            </div>

            <div className="space-y-4">
              {aiRecommendations.map((rec, index) => (
                <div key={index} className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h4 className="font-semibold text-lg text-gray-900 mb-1">{rec.title}</h4>
                      <div className="flex items-center space-x-4 text-sm text-gray-500">
                        <span className="capitalize">{rec.platform}</span>
                        <span>{rec.views} 观看</span>
                        <span className={`px-2 py-1 rounded ${difficultyColors[rec.difficulty]}`}>
                          {rec.difficulty}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-1">
                      <StarIcon className="w-5 h-5 text-yellow-400 fill-current" />
                      <span className="font-semibold text-yellow-600">{rec.aiScore}</span>
                    </div>
                  </div>

                  <div className="mb-3">
                    <div className="flex flex-wrap gap-2">
                      {rec.topics.map((topic, idx) => (
                        <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="bg-blue-50 border-l-4 border-blue-400 p-3 rounded">
                    <p className="text-sm text-blue-800">
                      <LightBulbIcon className="w-4 h-4 inline mr-1" />
                      AI推荐理由: {rec.reason}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-6 text-center">
              <h4 className="font-semibold text-green-800 mb-2">🌟 学习路径定制</h4>
              <p className="text-sm text-green-700">
                AI智能规划你的学习路径，根据当前水平和目标推荐最适合的学习内容
              </p>
              <button className="mt-3 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
                开始定制学习计划
              </button>
            </div>
          </div>
        )}

        {/* Social Impact Mode */}
        {analyzeMode === 'impact' && (
          <div className="space-y-6">
            <div className="text-center">
              <h3 className="text-xl font-semibold mb-2">🌍 教育公平与社会影响</h3>
              <p className="text-gray-600">通过技术创新推动教育公平，让优质教育资源触达更多人群</p>
            </div>

            {/* Accessibility Features */}
            <div>
              <h4 className="font-semibold mb-4">🔧 无障碍功能覆盖</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {socialImpact.accessibilityFeatures.map((feature, index) => (
                  <div key={index} className="bg-blue-50 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <h5 className="font-medium">{feature.feature}</h5>
                      <span className="text-sm font-semibold text-blue-600">{feature.coverage}</span>
                    </div>
                    <div className="text-sm text-gray-600">
                      已帮助 <span className="font-semibold text-blue-800">{feature.users.toLocaleString()}</span> 名用户
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Regional Impact */}
            <div>
              <h4 className="font-semibold mb-4">📍 区域教育改善情况</h4>
              <div className="space-y-4">
                {socialImpact.regions.map((region, index) => (
                  <div key={index} className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="flex justify-between items-center">
                      <div>
                        <h5 className="font-medium">{region.region}</h5>
                        <p className="text-sm text-gray-600">
                          服务用户: {region.users.toLocaleString()} 人
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-green-600">{region.improvement}</div>
                        <div className="text-xs text-green-700">学习效果提升</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Impact Statistics */}
            <div className="bg-gradient-to-br from-purple-100 to-pink-100 rounded-lg p-6">
              <h4 className="font-semibold mb-4 text-center">🎯 社会影响力数据</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
                <div>
                  <div className="text-3xl font-bold text-purple-600 mb-1">89%</div>
                  <div className="text-sm text-gray-600">用户学习效率提升</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-pink-600 mb-1">76%</div>
                  <div className="text-sm text-gray-600">教育资源获取便利性</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-indigo-600 mb-1">94%</div>
                  <div className="text-sm text-gray-600">用户满意度评分</div>
                </div>
              </div>
            </div>

            {/* Call to Action */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg p-6 text-white text-center">
              <h4 className="text-xl font-semibold mb-2">🚀 共建教育未来</h4>
              <p className="mb-4 opacity-90">
                加入我们的使命，用AI技术让每个人都能享受优质教育资源
              </p>
              <div className="flex justify-center space-x-4">
                <button className="px-6 py-2 bg-white text-blue-600 rounded-lg hover:bg-gray-100 transition-colors font-medium">
                  了解更多
                </button>
                <button className="px-6 py-2 bg-blue-800 text-white rounded-lg hover:bg-blue-900 transition-colors font-medium">
                  参与贡献
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default EducationAI;