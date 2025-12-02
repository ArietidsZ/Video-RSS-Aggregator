import { type Component, For, createSignal } from 'solid-js';

export interface Feed {
    id: string;
    title: string;
    icon?: string;
    unreadCount: number;
    lastUpdated: string;
}

interface FeedListProps {
    feeds: Feed[];
    selectedFeedId: string | null;
    onSelectFeed: (id: string) => void;
}

const FeedList: Component<FeedListProps> = (props) => {
    const [searchQuery, setSearchQuery] = createSignal('');

    const filteredFeeds = () => {
        const query = searchQuery().toLowerCase();
        return props.feeds.filter(feed => feed.title.toLowerCase().includes(query));
    };

    return (
        <div class="flex flex-col h-full">
            {/* Search Bar */}
            <div class="p-3 sticky top-0 z-10 bg-telegram-sidebar backdrop-blur-md">
                <div class="relative">
                    <input
                        type="text"
                        placeholder="Search"
                        class="w-full bg-[#182533] text-white placeholder-telegram-secondary px-4 py-2 pl-10 rounded-full focus:ring-2 focus:ring-telegram-primary outline-none transition-all text-sm"
                        value={searchQuery()}
                        onInput={(e) => setSearchQuery(e.currentTarget.value)}
                    />
                    <div class="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-telegram-secondary" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                        </svg>
                    </div>
                </div>
            </div>

            {/* Feed List */}
            <div class="flex-1 overflow-y-auto px-2 pb-2">
                <For each={filteredFeeds()}>
                    {(feed) => (
                        <div
                            class={`
                group flex items-center p-2.5 mb-0.5 rounded-lg cursor-pointer transition-colors
                ${props.selectedFeedId === feed.id ? 'bg-telegram-primary text-white' : 'hover:bg-white/5 text-telegram-text'}
              `}
                            onClick={() => props.onSelectFeed(feed.id)}
                        >
                            {/* Icon / Avatar */}
                            <div class={`
                w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold shrink-0 mr-3
                ${props.selectedFeedId === feed.id ? 'bg-white/20' : 'bg-gradient-to-br from-blue-500 to-cyan-500'}
              `}>
                                {feed.icon ? <img src={feed.icon} class="w-full h-full rounded-full" /> : feed.title.charAt(0).toUpperCase()}
                            </div>

                            {/* Content */}
                            <div class="flex-1 min-w-0">
                                <div class="flex justify-between items-baseline mb-0.5">
                                    <h3 class={`text-sm font-medium truncate ${props.selectedFeedId === feed.id ? 'text-white' : 'text-white'}`}>
                                        {feed.title}
                                    </h3>
                                    <span class={`text-xs ${props.selectedFeedId === feed.id ? 'text-white/70' : 'text-telegram-secondary'}`}>
                                        {feed.lastUpdated}
                                    </span>
                                </div>
                                <div class="flex justify-between items-center">
                                    <p class={`text-xs truncate ${props.selectedFeedId === feed.id ? 'text-white/80' : 'text-telegram-secondary'}`}>
                                        {/* Subtitle or last message preview could go here */}
                                        Latest updates
                                    </p>
                                    {feed.unreadCount > 0 && (
                                        <span class={`
                      ml-2 min-w-[1.25rem] h-5 px-1.5 rounded-full text-xs font-medium flex items-center justify-center
                      ${props.selectedFeedId === feed.id ? 'bg-white text-telegram-primary' : 'bg-telegram-primary text-white'}
                    `}>
                                            {feed.unreadCount}
                                        </span>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}
                </For>
            </div>
        </div>
    );
};

export default FeedList;
