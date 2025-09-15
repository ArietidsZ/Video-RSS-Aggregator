# Video RSS Frontend

Modern Next.js frontend for the Video RSS aggregator system.

## Features

- Real-time video feed display
- Platform filtering (Bilibili, Douyin, Kuaishou)
- Search and sorting functionality
- System metrics dashboard
- Configuration management
- RSS feed generation and download
- Responsive design with Tailwind CSS

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
cp .env.local.example .env.local
# Edit .env.local with your API URL
```

3. Start development server:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Configuration

Set the `NEXT_PUBLIC_API_URL` environment variable to point to your Rust backend server (default: `http://localhost:8080`).

## Development

- Uses TypeScript for type safety
- React Query for data fetching and caching
- Tailwind CSS for styling
- Lucide React for icons
- Automatic API error handling and retries

## Production

Build and deploy:
```bash
npm run build
npm start
```

The frontend automatically handles API connection failures and provides user feedback for all operations.
