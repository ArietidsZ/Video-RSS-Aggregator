use crate::{
    cache::*,
    database::*,
    extractor::*,
    resilience::*,
    transcription::*,
    types::*,
    Result,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tracing::info;

// Benchmark configuration
pub struct BenchmarkConfig {
    pub sample_size: usize,
    pub measurement_time: Duration,
    pub warm_up_time: Duration,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            sample_size: 100,
            measurement_time: Duration::from_secs(10),
            warm_up_time: Duration::from_secs(3),
        }
    }
}

// Cache performance benchmarks
pub fn bench_cache_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("cache_operations");
    group.sample_size(100);

    // Memory cache benchmarks
    group.bench_function("memory_cache_set", |b| {
        let cache = MemoryCache::new("bench".to_string());
        let test_data = generate_test_videos(100);

        b.to_async(&rt).iter(|| async {
            for (i, video) in test_data.iter().enumerate() {
                let key = format!("video_{}", i);
                let entry = CacheEntry::new(video.clone(), 3600);
                cache.set(&key, &entry, 3600).await.unwrap();
            }
        });
    });

    group.bench_function("memory_cache_get", |b| {
        let cache = MemoryCache::new("bench".to_string());
        let test_data = generate_test_videos(100);

        // Pre-populate cache
        rt.block_on(async {
            for (i, video) in test_data.iter().enumerate() {
                let key = format!("video_{}", i);
                let entry = CacheEntry::new(video.clone(), 3600);
                cache.set(&key, &entry, 3600).await.unwrap();
            }
        });

        b.to_async(&rt).iter(|| async {
            for i in 0..100 {
                let key = format!("video_{}", i);
                let _result: Option<CacheEntry<ExtractedVideo>> = cache.get(&key).await.unwrap();
            }
        });
    });

    group.finish();
}

// Database performance benchmarks
pub fn bench_database_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("database_operations");
    group.sample_size(50);

    // Setup test database
    let db = rt.block_on(async {
        let config = DatabaseConfig {
            url: "sqlite::memory:".to_string(),
            ..Default::default()
        };
        Database::new(config).await.unwrap()
    });

    group.bench_function("database_insert_video", |b| {
        let test_videos = generate_test_video_infos(100);

        b.to_async(&rt).iter(|| async {
            for video in &test_videos {
                db.insert_video(video).await.unwrap();
            }
        });
    });

    group.bench_function("database_query_by_platform", |b| {
        // Pre-populate database
        let test_videos = generate_test_video_infos(1000);
        rt.block_on(async {
            for video in &test_videos {
                db.insert_video(video).await.unwrap();
            }
        });

        b.to_async(&rt).iter(|| async {
            let _videos = db
                .get_videos_by_platform(Platform::Bilibili, 50, 0)
                .await
                .unwrap();
        });
    });

    group.finish();
}

// Circuit breaker performance benchmarks
pub fn bench_circuit_breaker(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("circuit_breaker");
    group.sample_size(100);

    group.bench_function("circuit_breaker_success", |b| {
        let cb = CircuitBreaker::new(
            "bench".to_string(),
            CircuitBreakerConfig::default(),
        );

        b.to_async(&rt).iter(|| async {
            cb.call(|| async { Ok::<i32, crate::error::VideoRssError>(42) })
                .await
                .unwrap()
        });
    });

    group.bench_function("circuit_breaker_failure", |b| {
        let cb = CircuitBreaker::new(
            "bench".to_string(),
            CircuitBreakerConfig::default(),
        );

        b.to_async(&rt).iter(|| async {
            let _ = cb
                .call(|| async {
                    Err::<i32, crate::error::VideoRssError>(crate::error::VideoRssError::Unknown(
                        "test".to_string(),
                    ))
                })
                .await;
        });
    });

    group.finish();
}

// RSS generation benchmarks
pub fn bench_rss_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("rss_generation");

    for size in [10, 50, 100, 500].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("generate_rss", size),
            size,
            |b, &size| {
                let videos = generate_test_video_infos(size);
                let generator = crate::rss::RssGenerator::default();

                b.to_async(&rt).iter(|| async {
                    let _rss = generator.generate_feed(black_box(&videos)).unwrap();
                });
            },
        );
    }

    group.finish();
}

// Concurrent request benchmarks
pub fn bench_concurrent_requests(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_requests");

    for concurrency in [10, 50, 100, 200].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_cache_operations", concurrency),
            concurrency,
            |b, &concurrency| {
                let cache = Arc::new(MemoryCache::new("bench".to_string()));

                b.to_async(&rt).iter(|| async {
                    let tasks: Vec<_> = (0..concurrency)
                        .map(|i| {
                            let cache = cache.clone();
                            tokio::spawn(async move {
                                let key = format!("key_{}", i);
                                let video = generate_test_videos(1)[0].clone();
                                let entry = CacheEntry::new(video, 3600);
                                cache.set(&key, &entry, 3600).await.unwrap();

                                let _result: Option<CacheEntry<ExtractedVideo>> = cache.get(&key).await.unwrap();
                            })
                        })
                        .collect();

                    for task in tasks {
                        task.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// Load testing utilities
pub struct LoadTestConfig {
    pub duration: Duration,
    pub target_rps: u64,
    pub max_concurrent: usize,
}

pub struct LoadTestResult {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time: Duration,
    pub min_response_time: Duration,
    pub max_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub actual_rps: f64,
}

pub async fn run_load_test<F, Fut>(
    config: LoadTestConfig,
    request_fn: F,
) -> Result<LoadTestResult>
where
    F: Fn() -> Fut + Send + Sync + Clone + 'static,
    Fut: std::future::Future<Output = Result<Duration>> + Send + 'static,
{
    use tokio::sync::Semaphore;
    use tokio::time::{sleep, Instant};

    let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
    let start_time = Instant::now();
    let mut handles = Vec::new();
    let mut response_times = Vec::new();
    let mut successful_requests = 0u64;
    let mut failed_requests = 0u64;

    let interval = Duration::from_nanos(1_000_000_000 / config.target_rps);
    let mut next_request_time = start_time;

    while start_time.elapsed() < config.duration {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let request_fn = request_fn.clone();

        let handle = tokio::spawn(async move {
            let _permit = permit;
            let start = Instant::now();
            match request_fn().await {
                Ok(duration) => (true, duration),
                Err(_) => (false, start.elapsed()),
            }
        });

        handles.push(handle);

        next_request_time += interval;
        let now = Instant::now();
        if next_request_time > now {
            sleep(next_request_time - now).await;
        }
    }

    // Collect results
    for handle in handles {
        match handle.await {
            Ok((success, duration)) => {
                response_times.push(duration);
                if success {
                    successful_requests += 1;
                } else {
                    failed_requests += 1;
                }
            }
            Err(_) => {
                failed_requests += 1;
            }
        }
    }

    // Calculate statistics
    response_times.sort();
    let total_requests = successful_requests + failed_requests;
    let avg_response_time = if !response_times.is_empty() {
        response_times.iter().sum::<Duration>() / response_times.len() as u32
    } else {
        Duration::default()
    };

    let p95_index = (response_times.len() as f64 * 0.95) as usize;
    let p99_index = (response_times.len() as f64 * 0.99) as usize;

    Ok(LoadTestResult {
        total_requests,
        successful_requests,
        failed_requests,
        avg_response_time,
        min_response_time: response_times.first().copied().unwrap_or_default(),
        max_response_time: response_times.last().copied().unwrap_or_default(),
        p95_response_time: response_times.get(p95_index).copied().unwrap_or_default(),
        p99_response_time: response_times.get(p99_index).copied().unwrap_or_default(),
        actual_rps: total_requests as f64 / config.duration.as_secs_f64(),
    })
}

// Performance regression tests
pub async fn run_performance_regression_tests() -> Result<()> {
    info!("Running performance regression tests...");

    // Cache performance test
    let cache_result = time_operation(|| async {
        let cache = MemoryCache::new("test".to_string());
        for i in 0..1000 {
            let key = format!("key_{}", i);
            let video = generate_test_videos(1)[0].clone();
            let entry = CacheEntry::new(video, 3600);
            cache.set(&key, &entry, 3600).await.unwrap();
        }
    })
    .await;

    assert!(
        cache_result < Duration::from_millis(100),
        "Cache performance regression: {}ms > 100ms",
        cache_result.as_millis()
    );

    // RSS generation performance test
    let rss_result = time_operation(|| async {
        let videos = generate_test_video_infos(100);
        let generator = crate::rss::RssGenerator::default();
        let _rss = generator.generate_feed(&videos).unwrap();
    })
    .await;

    assert!(
        rss_result < Duration::from_millis(50),
        "RSS generation performance regression: {}ms > 50ms",
        rss_result.as_millis()
    );

    info!("Performance regression tests passed");
    Ok(())
}

// Stress testing
pub async fn run_stress_tests() -> Result<()> {
    info!("Running stress tests...");

    // High concurrency cache test
    let cache = Arc::new(MemoryCache::new("stress".to_string()));
    let tasks: Vec<_> = (0..1000)
        .map(|i| {
            let cache = cache.clone();
            tokio::spawn(async move {
                for j in 0..100 {
                    let key = format!("key_{}_{}", i, j);
                    let video = generate_test_videos(1)[0].clone();
                    let entry = CacheEntry::new(video, 3600);
                    cache.set(&key, &entry, 3600).await.unwrap();

                    let _result: Option<CacheEntry<ExtractedVideo>> = cache.get(&key).await.unwrap();
                }
            })
        })
        .collect();

    for task in tasks {
        task.await.unwrap();
    }

    info!("Stress tests completed successfully");
    Ok(())
}

// Utility functions
async fn time_operation<F, Fut>(operation: F) -> Duration
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let start = Instant::now();
    operation().await;
    start.elapsed()
}

fn generate_test_videos(count: usize) -> Vec<ExtractedVideo> {
    (0..count)
        .map(|i| ExtractedVideo {
            id: format!("test_{}", i),
            title: format!("Test Video {}", i),
            platform: Platform::Bilibili,
            author: format!("Author {}", i),
            url: format!("https://example.com/video_{}", i),
            description: format!("Description for video {}", i),
            view_count: Some(1000 + i as u64),
            like_count: Some(100 + i as u64),
            duration: Some(format!("{}:00", i % 60)),
            upload_date: format!("2024-01-{:02}", (i % 30) + 1),
            thumbnail: Some(format!("https://example.com/thumb_{}.jpg", i)),
            tags: vec![format!("tag_{}", i), "test".to_string()],
            data_source: "benchmark".to_string(),
            legal_compliance: "Test Data".to_string(),
            extraction_method: "Generated".to_string(),
            extracted_at: chrono::Utc::now(),
        })
        .collect()
}

fn generate_test_video_infos(count: usize) -> Vec<VideoInfo> {
    (0..count)
        .map(|i| VideoInfo {
            id: format!("test_{}", i),
            title: format!("Test Video {}", i),
            description: format!("Description for video {}", i),
            url: format!("https://example.com/video_{}", i),
            author: format!("Author {}", i),
            upload_date: chrono::Utc::now(),
            duration: Some(300 + i as u64),
            view_count: 1000 + i as u64,
            like_count: 100 + i as u64,
            comment_count: 10 + i as u64,
            tags: vec![format!("tag_{}", i), "test".to_string()],
            thumbnail_url: Some(format!("https://example.com/thumb_{}.jpg", i)),
            platform: Platform::Bilibili,
            transcription: None,
        })
        .collect()
}

// Criterion benchmark groups
criterion_group!(
    benches,
    bench_cache_operations,
    bench_database_operations,
    bench_circuit_breaker,
    bench_rss_generation,
    bench_concurrent_requests
);

criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_regression() {
        run_performance_regression_tests().await.unwrap();
    }

    #[tokio::test]
    async fn test_stress() {
        run_stress_tests().await.unwrap();
    }

    #[tokio::test]
    async fn test_load_testing() {
        let config = LoadTestConfig {
            duration: Duration::from_secs(5),
            target_rps: 100,
            max_concurrent: 50,
        };

        let result = run_load_test(config, || async {
            // Simulate some work
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(Duration::from_millis(10))
        })
        .await
        .unwrap();

        assert!(result.successful_requests > 0);
        assert!(result.actual_rps > 0.0);
    }
}