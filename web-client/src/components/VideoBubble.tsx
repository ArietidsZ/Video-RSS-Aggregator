import { type Component, Show } from 'solid-js';
import VideoPlayer from './VideoPlayer';

export interface VideoItem {
    id: string;
    title: string;
    url: string;
    summary?: string;
    publishedAt: string;
    feedTitle: string;
}

interface VideoBubbleProps {
    video: VideoItem;
}

const VideoBubble: Component<VideoBubbleProps> = (props) => {
    // const [isExpanded, setIsExpanded] = createSignal(false); // Unused for now

    return (
        <div class="flex flex-col items-start mb-6 max-w-3xl">
            {/* Feed Title (Sender Name) */}
            <div class="ml-4 mb-1 text-sm font-medium text-telegram-primary">
                {props.video.feedTitle}
            </div>

            {/* Bubble Container */}
            <div class="bg-telegram-card rounded-2xl rounded-tl-none p-3 shadow-sm w-full overflow-hidden">

                {/* Video Player */}
                <div class="rounded-xl overflow-hidden bg-black aspect-video mb-3">
                    <VideoPlayer src={props.video.url} title={props.video.title} />
                </div>

                {/* Content */}
                <div class="px-1">
                    <h3 class="text-lg font-semibold text-white mb-2 leading-tight">
                        {props.video.title}
                    </h3>

                    <Show when={props.video.summary}>
                        <div class="text-telegram-text text-sm leading-relaxed whitespace-pre-wrap mb-2">
                            {props.video.summary}
                        </div>
                    </Show>

                    {/* Metadata Footer */}
                    <div class="flex justify-end items-center mt-2 space-x-2">
                        <span class="text-xs text-telegram-secondary">
                            {props.video.publishedAt}
                        </span>
                        {/* Read More / Expand (Optional) */}
                        {/* <button 
              class="text-xs text-telegram-primary hover:underline"
              onClick={() => setIsExpanded(!isExpanded())}
            >
              {isExpanded() ? 'Show Less' : 'Read More'}
            </button> */}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default VideoBubble;
