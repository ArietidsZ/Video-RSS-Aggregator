#!/usr/bin/env python3
import json
import requests
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    try:
        os.system('chcp 65001 >nul 2>&1')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Test the transcription API
response = requests.get("http://localhost:8001/api/recommendations/bilibili?limit=1&include_transcription=true")
data = response.json()

video = data['videos'][0]
t = video['transcription']

print('=== UPDATED TRANSCRIPTION RESULT ===')
print('Video Title:', video['title'][:60] + '...')
print('Status:', t['status'])
print('Transcriber:', t['model_info']['transcriber'])
print('Summarizer:', t['model_info']['summarizer'])
print()
print('📄 Paragraph Summary:')
ps = t.get('paragraph_summary', 'MISSING')
print(f'   Length: {len(ps)} chars')
if ps != 'MISSING':
    print(f'   Content: {ps}')
print()
print('📝 Sentence Subtitle:')
ss = t.get('sentence_subtitle', 'MISSING')
print(f'   Length: {len(ss)} chars')
if ss != 'MISSING':
    print(f'   Content: {ss}')
print()
print('📰 Full Transcript:')
ft = t.get('transcript', 'MISSING')
print(f'   Length: {len(ft)} chars')
if ft != 'MISSING':
    print(f'   Preview: {ft[:100]}...')
print()
print('✅ SUCCESS:', 'YES' if (len(ps) > 10 and len(ss) > 5) else 'NO')