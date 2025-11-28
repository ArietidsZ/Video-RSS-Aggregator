import { createSignal } from 'solid-js';
import VideoPlayer from './components/VideoPlayer';
import FeedList from './components/FeedList';

function App() {
  const [currentVideo, setCurrentVideo] = createSignal<{ url: string, title: string } | null>(null);
  const [items] = createSignal([
    {
      id: '1',
      title: 'Sample Video: TED Talk',
      url: 'https://www.youtube.com/watch?v=d4eDWc8g0e0', // Example
      description: 'This is a sample video description to demonstrate the UI layout.'
    },
    // Add more mock items or fetch from API
  ]);

  // In real app: fetch items from Gateway API
  // onMount(async () => {
  //   const res = await fetch('http://localhost:8080/api/feeds/items');
  //   const data = await res.json();
  //   setItems(data);
  // });

  return (
    <div class="min-h-screen bg-gray-900 text-white">
      <header class="bg-gray-800 p-4 shadow-md">
        <h1 class="text-2xl font-bold text-blue-400">Video RSS Aggregator</h1>
      </header>

      <main class="container mx-auto py-8">
        {currentVideo() && (
          <div class="mb-8 p-4">
            <div class="aspect-video bg-black rounded-lg overflow-hidden shadow-2xl max-w-4xl mx-auto">
              <VideoPlayer src={currentVideo()!.url} title={currentVideo()!.title} />
            </div>
            <h2 class="text-xl font-semibold mt-4 text-center">{currentVideo()!.title}</h2>
          </div>
        )}

        <h2 class="text-xl font-semibold px-4 mb-4">Latest Videos</h2>
        <FeedList
          items={items()}
          onPlay={(url, title) => setCurrentVideo({ url, title })}
        />
      </main>
    </div>
  );
}

export default App;
