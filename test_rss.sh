#!/bin/bash

echo "Testing RSS Feed Generation"
echo "============================"

# Test server health
echo -e "\n1. Testing server health..."
curl -s http://localhost:8080/health | jq . || echo "Server not responding"

# Test getting videos from Bilibili
echo -e "\n2. Fetching Bilibili videos..."
curl -s "http://localhost:8080/videos?platforms=bilibili&limit=5" | jq '.[] | {title: .title, author: .author, views: .view_count}' || echo "Failed to fetch videos"

# Test RSS generation
echo -e "\n3. Generating RSS feed..."
curl -s -X POST "http://localhost:8080/rss/generate" \
  -H "Content-Type: application/json" \
  -d '{"platforms": ["bilibili"]}' > test_rss.xml

if [ -f test_rss.xml ]; then
    echo "RSS feed saved to test_rss.xml"
    echo "First 500 characters of RSS:"
    head -c 500 test_rss.xml
    echo -e "\n..."
else
    echo "Failed to generate RSS"
fi

# Test metrics
echo -e "\n4. Checking system metrics..."
curl -s http://localhost:8080/stats | jq . || echo "Failed to get stats"

echo -e "\nTest complete!"