import { createSignal, For } from 'solid-js';
import Layout from './components/Layout';
import FeedList, { type Feed } from './components/FeedList';
import VideoBubble, { type VideoItem } from './components/VideoBubble';

function App() {
  const [selectedFeedId, setSelectedFeedId] = createSignal<string | null>('1');

  // Mock Feeds
  const [feeds] = createSignal<Feed[]>([
    { id: '1', title: 'Tech News Daily', unreadCount: 3, lastUpdated: '10:42 AM' },
    { id: '2', title: 'Rust Programming', unreadCount: 0, lastUpdated: 'Yesterday' },
    { id: '3', title: 'AI Research', unreadCount: 12, lastUpdated: 'Mon' },
    { id: '4', title: 'Cooking Channel', unreadCount: 0, lastUpdated: 'Sun' },
  ]);

  // Mock Videos (Messages)
  const [videos] = createSignal<VideoItem[]>([
    {
      id: '101',
      title: 'The Future of AI Agents',
      url: 'https://www.w3schools.com/html/mov_bbb.mp4', // Sample video
      feedTitle: 'Tech News Daily',
      publishedAt: '10:42 AM',
      summary: 'This video discusses the rapid evolution of AI agents, focusing on their ability to perform complex tasks autonomously. Key points include:\n• Shift from chat-based to agentic workflows\n• Integration with external tools\n• Future implications for software development'
    },
    {
      id: '102',
      title: 'Understanding Rust Ownership',
      url: 'https://www.w3schools.com/html/movie.mp4',
      feedTitle: 'Rust Programming',
      publishedAt: 'Yesterday',
      summary: 'A deep dive into Rust\'s ownership system. The speaker explains how move semantics and borrowing work under the hood to ensure memory safety without a garbage collector.'
    }
  ]);

  return (
    <Layout
      sidebar={
        <FeedList
          feeds={feeds()}
          selectedFeedId={selectedFeedId()}
          onSelectFeed={setSelectedFeedId}
        />
      }
    >
      <div class="flex flex-col justify-end min-h-full pb-4">
        {/* Date Separator Example */}
        <div class="flex justify-center mb-6">
          <span class="bg-black/20 text-white/60 text-xs px-3 py-1 rounded-full backdrop-blur-sm">
            Today
          </span>
        </div>

        <For each={videos()}>
          {(video) => <VideoBubble video={video} />}
        </For>
      </div>
    </Layout>
  );
}

export default App;
