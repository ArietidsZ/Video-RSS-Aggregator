import { For } from 'solid-js';

interface FeedItem {
    id: string;
    title: string;
    url: string;
    description: string;
}

interface FeedListProps {
    items: FeedItem[];
    onPlay: (url: string, title: string) => void;
}

export default function FeedList(props: FeedListProps) {
    return (
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
            <For each={props.items}>
                {(item) => (
                    <div class="bg-gray-800 rounded-lg overflow-hidden shadow-lg hover:bg-gray-700 transition-colors">
                        <div class="p-4">
                            <h3 class="text-lg font-bold mb-2 truncate" title={item.title}>{item.title}</h3>
                            <p class="text-gray-400 text-sm mb-4 line-clamp-3">{item.description}</p>
                            <button
                                class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded w-full"
                                onClick={() => props.onPlay(item.url, item.title)}
                            >
                                Play Video
                            </button>
                        </div>
                    </div>
                )}
            </For>
        </div>
    );
}
