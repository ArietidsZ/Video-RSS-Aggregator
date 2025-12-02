import { type Component, type JSX, Show, createSignal } from 'solid-js';

interface LayoutProps {
    sidebar: JSX.Element;
    children: JSX.Element;
}

const Layout: Component<LayoutProps> = (props) => {
    const [isSidebarOpen, setIsSidebarOpen] = createSignal(true);

    return (
        <div class="flex h-screen overflow-hidden bg-telegram-bg text-telegram-text">
            {/* Mobile Sidebar Overlay */}
            <Show when={isSidebarOpen()}>
                <div
                    class="fixed inset-0 z-20 bg-black/50 lg:hidden"
                    onClick={() => setIsSidebarOpen(false)}
                />
            </Show>

            {/* Sidebar */}
            <aside
                class={`
          fixed inset-y-0 left-0 z-30 w-80 transform bg-telegram-sidebar border-r border-black/10 transition-transform duration-300 ease-in-out lg:static lg:translate-x-0
          ${isSidebarOpen() ? 'translate-x-0' : '-translate-x-full'}
        `}
            >
                <div class="flex h-full flex-col">
                    {/* Sidebar Header (Search/Branding could go here) */}
                    <div class="flex h-14 items-center justify-between px-4 border-b border-white/5 shrink-0">
                        <span class="font-semibold text-white">Feeds</span>
                        <button
                            class="lg:hidden p-1 rounded hover:bg-white/10"
                            onClick={() => setIsSidebarOpen(false)}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>

                    {/* Sidebar Content (Scrollable) */}
                    <div class="flex-1 overflow-y-auto custom-scrollbar">
                        {props.sidebar}
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main class="flex-1 flex flex-col min-w-0 overflow-hidden relative">
                {/* Mobile Header */}
                <div class="flex items-center h-14 px-4 border-b border-white/5 lg:hidden shrink-0">
                    <button
                        class="p-2 -ml-2 rounded-lg hover:bg-white/5 text-telegram-secondary"
                        onClick={() => setIsSidebarOpen(true)}
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
                        </svg>
                    </button>
                    <span class="ml-3 font-semibold">Video RSS</span>
                </div>

                {/* Main Scrollable Area */}
                <div class="flex-1 overflow-y-auto custom-scrollbar p-4 lg:p-6">
                    <div class="mx-auto max-w-4xl">
                        {props.children}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default Layout;
