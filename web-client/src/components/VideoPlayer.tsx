interface VideoPlayerProps {
    src: string;
    title: string;
}

export default function VideoPlayer(props: VideoPlayerProps) {
    return (
        <div class="w-full h-full">
            <video
                controls
                class="w-full h-full"
                src={props.src}
                title={props.title}
            >
                Your browser does not support the video tag.
            </video>
        </div>
    );
}
