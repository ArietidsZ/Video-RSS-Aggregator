#!/usr/bin/env python3
"""
Test RSS content with fallback videos
"""

import asyncio
import sys
import os

# Set UTF-8 encoding
if sys.platform.startswith('win'):
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

async def test_rss():
    # Create fallback test videos
    test_videos = [
        {
            'bvid': 'BV1xx411c7BF',
            'title': 'AI时代的程序员：ChatGPT和GitHub Copilot使用技巧',
            'author': 'Tech Creator',
            'view': 50000,
            'danmaku': 200,
            'like': 1500,
            'coin': 800,
            'duration': '15:30',
            'pic': 'https://example.com/cover1.jpg',
            'full_transcription': {
                'transcript': '今天我们来讨论AI工具如何改变程序员的工作流程。ChatGPT可以帮助我们理解复杂的代码逻辑，生成文档，甚至协助调试。而GitHub Copilot则能实时提供代码建议，大幅提升编码效率。',
                'paragraph_summary': 'AI工具正在革新软件开发：ChatGPT助力代码理解和文档生成，GitHub Copilot提供智能代码补全，两者结合可将开发效率提升300%。关键在于学会提问技巧和理解AI的局限性。',
                'sentence_subtitle': 'AI赋能开发者，效率提升300%',
                'status': 'success'
            }
        },
        {
            'bvid': 'BV1yy4y1z7Pf',
            'title': '深度学习入门：从零开始构建神经网络',
            'author': 'AI教程',
            'view': 35000,
            'danmaku': 150,
            'like': 1200,
            'coin': 600,
            'duration': '20:45',
            'pic': 'https://example.com/cover2.jpg',
            'full_transcription': {
                'transcript': '神经网络是深度学习的基础。今天我们从最简单的感知器开始，逐步构建多层神经网络。首先理解前向传播，然后是反向传播算法的数学原理。',
                'paragraph_summary': '深度学习入门教程：从感知器到多层神经网络，详解前向传播和反向传播的数学原理。使用Python和NumPy从零实现，无需深度学习框架，适合初学者理解底层原理。',
                'sentence_subtitle': '从零构建神经网络，掌握深度学习精髓',
                'status': 'success'
            }
        },
        {
            'bvid': 'BV1Qb421x7mj',
            'title': '量子计算机原理解析：未来已来',
            'author': '科技前沿',
            'view': 28000,
            'danmaku': 100,
            'like': 900,
            'coin': 400,
            'duration': '18:20',
            'pic': 'https://example.com/cover3.jpg',
            'full_transcription': {
                'transcript': '量子计算利用量子叠加和纠缠特性，能够并行处理海量信息。与传统计算机的二进制不同，量子比特可以同时表示0和1的叠加态。',
                'paragraph_summary': '量子计算机通过量子叠加和纠缠实现指数级计算加速。详解量子比特、量子门和量子算法，探讨其在密码学、药物研发和人工智能领域的革命性应用前景。',
                'sentence_subtitle': '量子计算：开启计算新纪元',
                'status': 'success'
            }
        }
    ]

    # Generate RSS XML
    from xml.etree import ElementTree as ET
    from datetime import datetime

    rss = ET.Element("rss", version="2.0")
    rss.set("xmlns:content", "http://purl.org/rss/1.0/modules/content/")

    channel = ET.SubElement(rss, "channel")

    # Channel metadata
    ET.SubElement(channel, "title").text = "AI Video Digest - Test Content"
    ET.SubElement(channel, "description").text = "Ultra-optimized AI-powered video content with transcription"
    ET.SubElement(channel, "link").text = "http://111.186.3.124:8000"
    ET.SubElement(channel, "lastBuildDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Add video items with AI transcription
    for video in test_videos:
        item = ET.SubElement(channel, "item")

        ET.SubElement(item, "title").text = video['title']
        ET.SubElement(item, "link").text = f"https://www.bilibili.com/video/{video['bvid']}"
        ET.SubElement(item, "author").text = video['author']

        # Add AI-generated content
        trans = video['full_transcription']

        # Description with AI summary
        description = f"""
🤖 AI摘要：{trans['paragraph_summary']}

📝 一句话总结：{trans['sentence_subtitle']}

📊 视频数据：
• 播放量：{video['view']:,}
• 点赞：{video['like']:,}
• 投币：{video['coin']:,}
• 弹幕：{video['danmaku']:,}
• 时长：{video['duration']}
        """
        ET.SubElement(item, "description").text = description.strip()

        # Full content with HTML formatting
        content = f"""
<h3>{video['title']}</h3>
<p><strong>作者：</strong>{video['author']}</p>

<h4>🤖 AI智能摘要</h4>
<p>{trans['paragraph_summary']}</p>

<h4>📝 核心要点</h4>
<p>{trans['sentence_subtitle']}</p>

<h4>📄 转录内容节选</h4>
<blockquote>{trans['transcript']}</blockquote>

<h4>📊 互动数据</h4>
<ul>
<li>播放量：{video['view']:,}</li>
<li>点赞数：{video['like']:,}</li>
<li>投币数：{video['coin']:,}</li>
<li>弹幕数：{video['danmaku']:,}</li>
<li>视频时长：{video['duration']}</li>
</ul>

<p><small>🔄 处理状态：{trans['status']} | ⏰ 生成时间：{datetime.now().isoformat()}</small></p>
        """
        content_elem = ET.SubElement(item, "{http://purl.org/rss/1.0/modules/content/}encoded")
        content_elem.text = content.strip()

        # Add publication date
        ET.SubElement(item, "pubDate").text = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")

    # Convert to string with pretty formatting
    rss_xml = ET.tostring(rss, encoding='unicode')

    # Pretty print the XML
    import xml.dom.minidom
    dom = xml.dom.minidom.parseString(rss_xml)
    pretty_xml = dom.toprettyxml(indent="  ")

    print("=" * 80)
    print("RSS FEED CONTENT WITH AI TRANSCRIPTION")
    print("=" * 80)
    print(pretty_xml)

    # Save to file
    with open('/tmp/test_rss_output.xml', 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

    print("\n" + "=" * 80)
    print("RSS feed saved to: /tmp/test_rss_output.xml")
    print("This demonstrates the RSS content structure with AI-generated summaries")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_rss())