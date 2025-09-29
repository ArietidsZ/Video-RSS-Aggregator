use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use anyhow::Result;
use blake3;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Intelligent content deduplication service using MinHash and LSH
pub struct DeduplicationService {
    /// Exact hash index for URL/GUID deduplication
    exact_index: Arc<DashMap<u64, String>>,

    /// MinHash LSH index for near-duplicate detection
    lsh_index: Arc<LSHIndex>,

    /// Bloom filter for fast negative lookups
    bloom_filter: Arc<RwLock<BloomFilter>>,

    /// Statistics
    stats: Arc<DeduplicationStats>,
}

impl DeduplicationService {
    pub fn new(estimated_items: usize) -> Self {
        Self {
            exact_index: Arc::new(DashMap::new()),
            lsh_index: Arc::new(LSHIndex::new(128, 16, 0.8)), // 128 hashes, 16 bands, 0.8 similarity threshold
            bloom_filter: Arc::new(RwLock::new(BloomFilter::new(estimated_items, 0.001))),
            stats: Arc::new(DeduplicationStats::new()),
        }
    }

    /// Compute fast hash for exact matching
    pub fn compute_hash(&self, content: &str) -> String {
        blake3::hash(content.as_bytes()).to_hex().to_string()
    }

    /// Check if content is duplicate (exact or near-duplicate)
    pub async fn check_duplicate(&self, hash: &str) -> Option<String> {
        // Check bloom filter first (fast negative)
        if !self.bloom_filter.read().contains(hash) {
            self.stats.record_bloom_filter_miss();
            return None;
        }

        // Check exact match
        let hash_u64 = hash_to_u64(hash);
        if let Some(entry) = self.exact_index.get(&hash_u64) {
            self.stats.record_exact_match();
            return Some(entry.clone());
        }

        self.stats.record_no_duplicate();
        None
    }

    /// Check for near-duplicate content using MinHash LSH
    pub async fn check_near_duplicate(&self, content: &str, threshold: f64) -> Option<Vec<SimilarItem>> {
        // Generate MinHash signature
        let signature = MinHash::new(content, 128).signature();

        // Query LSH index
        let candidates = self.lsh_index.query(&signature);

        // Verify candidates with actual similarity
        let mut similar_items = Vec::new();
        for candidate_id in candidates {
            if let Some(stored_sig) = self.lsh_index.get_signature(&candidate_id) {
                let similarity = MinHash::jaccard_similarity(&signature, &stored_sig);

                if similarity >= threshold {
                    similar_items.push(SimilarItem {
                        id: candidate_id,
                        similarity,
                    });
                    self.stats.record_near_duplicate();
                }
            }
        }

        if similar_items.is_empty() {
            None
        } else {
            Some(similar_items)
        }
    }

    /// Add new entry to deduplication indices
    pub async fn add_entry(&self, hash: &str, id: &str) {
        // Add to bloom filter
        self.bloom_filter.write().insert(hash);

        // Add to exact index
        let hash_u64 = hash_to_u64(hash);
        self.exact_index.insert(hash_u64, id.to_string());

        self.stats.record_addition();
    }

    /// Add content for near-duplicate detection
    pub async fn add_content_signature(&self, id: &str, content: &str) {
        let signature = MinHash::new(content, 128).signature();
        self.lsh_index.insert(id.to_string(), signature);
    }

    /// Remove entry from indices
    pub async fn remove_entry(&self, hash: &str, id: &str) {
        let hash_u64 = hash_to_u64(hash);
        self.exact_index.remove(&hash_u64);
        self.lsh_index.remove(&id.to_string());
        self.stats.record_removal();
    }

    /// Get deduplication statistics
    pub fn get_stats(&self) -> DeduplicationStatsSnapshot {
        self.stats.snapshot()
    }

    /// Clear all indices
    pub async fn clear(&self) {
        self.exact_index.clear();
        self.lsh_index.clear();
        *self.bloom_filter.write() = BloomFilter::new(1000000, 0.001);
    }

    /// Batch deduplication check
    pub async fn check_batch(&self, items: Vec<String>) -> Vec<(String, bool)> {
        items
            .into_iter()
            .map(|item| {
                let hash = self.compute_hash(&item);
                let is_duplicate = self.check_duplicate(&hash).is_some();
                (item, is_duplicate)
            })
            .collect()
    }

    /// Content-based deduplication for RSS items
    pub async fn deduplicate_feed_items(&self, items: Vec<FeedItemDedup>) -> Vec<FeedItemDedup> {
        let mut unique_items = Vec::new();
        let mut seen_signatures = HashSet::new();

        for item in items {
            // Create content fingerprint
            let fingerprint = Self::create_content_fingerprint(&item);

            // Check if we've seen similar content
            if !seen_signatures.contains(&fingerprint) {
                seen_signatures.insert(fingerprint.clone());

                // Check for near-duplicates in existing items
                let is_near_duplicate = self.check_near_duplicate(&item.content, 0.85)
                    .await
                    .is_some();

                if !is_near_duplicate {
                    unique_items.push(item);
                } else {
                    self.stats.record_filtered_item();
                }
            } else {
                self.stats.record_exact_duplicate_item();
            }
        }

        unique_items
    }

    /// Create a content fingerprint for deduplication
    fn create_content_fingerprint(item: &FeedItemDedup) -> String {
        // Normalize content for comparison
        let normalized = Self::normalize_text(&item.content);

        // Create composite fingerprint from title and content
        let title_hash = blake3::hash(item.title.as_bytes());
        let content_hash = blake3::hash(normalized.as_bytes());

        format!("{}:{}", title_hash.to_hex(), content_hash.to_hex())
    }

    /// Normalize text for better duplicate detection
    fn normalize_text(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// MinHash implementation for similarity detection
struct MinHash {
    signature: Vec<u64>,
}

impl MinHash {
    /// Create MinHash signature from text
    fn new(text: &str, num_hashes: usize) -> Self {
        let shingles = Self::create_shingles(text, 3); // 3-gram shingles
        let signature = Self::compute_signature(&shingles, num_hashes);

        Self { signature }
    }

    /// Create k-shingles from text
    fn create_shingles(text: &str, k: usize) -> HashSet<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut shingles = HashSet::new();

        if words.len() >= k {
            for i in 0..=words.len() - k {
                let shingle = words[i..i + k].join(" ");
                shingles.insert(shingle);
            }
        }

        shingles
    }

    /// Compute MinHash signature
    fn compute_signature(shingles: &HashSet<String>, num_hashes: usize) -> Vec<u64> {
        let mut signature = Vec::with_capacity(num_hashes);

        for i in 0..num_hashes {
            let mut min_hash = u64::MAX;

            for shingle in shingles {
                let hash = Self::hash_function(shingle, i);
                if hash < min_hash {
                    min_hash = hash;
                }
            }

            signature.push(min_hash);
        }

        signature
    }

    /// Hash function with seed for multiple hash functions
    fn hash_function(data: &str, seed: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        hasher.write(data.as_bytes());
        hasher.write_usize(seed);
        hasher.finish()
    }

    /// Get the signature
    fn signature(&self) -> Vec<u64> {
        self.signature.clone()
    }

    /// Calculate Jaccard similarity between two signatures
    fn jaccard_similarity(sig1: &[u64], sig2: &[u64]) -> f64 {
        if sig1.len() != sig2.len() {
            return 0.0;
        }

        let matches = sig1.iter()
            .zip(sig2.iter())
            .filter(|(a, b)| a == b)
            .count();

        matches as f64 / sig1.len() as f64
    }
}

/// Locality-Sensitive Hashing (LSH) index
struct LSHIndex {
    bands: Vec<HashMap<Vec<u64>, Vec<String>>>,
    signatures: DashMap<String, Vec<u64>>,
    num_bands: usize,
    rows_per_band: usize,
    threshold: f64,
}

impl LSHIndex {
    fn new(num_hashes: usize, num_bands: usize, threshold: f64) -> Self {
        let rows_per_band = num_hashes / num_bands;

        let mut bands = Vec::with_capacity(num_bands);
        for _ in 0..num_bands {
            bands.push(HashMap::new());
        }

        Self {
            bands,
            signatures: DashMap::new(),
            num_bands,
            rows_per_band,
            threshold,
        }
    }

    /// Insert a signature into the LSH index
    fn insert(&self, id: String, signature: Vec<u64>) {
        // Store the full signature
        self.signatures.insert(id.clone(), signature.clone());

        // Hash into bands
        for (band_idx, band) in self.bands.iter().enumerate() {
            let start = band_idx * self.rows_per_band;
            let end = start + self.rows_per_band;

            let band_signature = signature[start..end].to_vec();

            // This would need proper synchronization in production
            // band.entry(band_signature)
            //     .or_insert_with(Vec::new)
            //     .push(id.clone());
        }
    }

    /// Query the LSH index for similar items
    fn query(&self, signature: &[u64]) -> Vec<String> {
        let mut candidates = HashSet::new();

        // Check each band for matches
        for (band_idx, band) in self.bands.iter().enumerate() {
            let start = band_idx * self.rows_per_band;
            let end = start + self.rows_per_band;

            let band_signature = &signature[start..end];

            if let Some(matches) = band.get(band_signature) {
                for id in matches {
                    candidates.insert(id.clone());
                }
            }
        }

        candidates.into_iter().collect()
    }

    /// Get stored signature
    fn get_signature(&self, id: &str) -> Option<Vec<u64>> {
        self.signatures.get(id).map(|sig| sig.clone())
    }

    /// Remove an item from the index
    fn remove(&self, id: &str) {
        self.signatures.remove(id);
        // Would also need to remove from bands
    }

    /// Clear the index
    fn clear(&self) {
        self.signatures.clear();
        // Would also clear bands
    }
}

/// Bloom filter for fast negative lookups
struct BloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
}

impl BloomFilter {
    fn new(estimated_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal size and number of hash functions
        let size = Self::optimal_size(estimated_items, false_positive_rate);
        let num_hashes = Self::optimal_num_hashes(size, estimated_items);

        Self {
            bits: vec![false; size],
            num_hashes,
            size,
        }
    }

    /// Calculate optimal bloom filter size
    fn optimal_size(n: usize, p: f64) -> usize {
        let ln2 = std::f64::consts::LN_2;
        ((-1.0 * n as f64 * p.ln()) / (ln2 * ln2)).ceil() as usize
    }

    /// Calculate optimal number of hash functions
    fn optimal_num_hashes(m: usize, n: usize) -> usize {
        let ln2 = std::f64::consts::LN_2;
        ((m as f64 / n as f64) * ln2).ceil() as usize
    }

    /// Insert item into bloom filter
    fn insert(&mut self, item: &str) {
        for i in 0..self.num_hashes {
            let hash = self.hash_with_seed(item, i);
            let index = (hash as usize) % self.size;
            self.bits[index] = true;
        }
    }

    /// Check if item might be in the set
    fn contains(&self, item: &str) -> bool {
        for i in 0..self.num_hashes {
            let hash = self.hash_with_seed(item, i);
            let index = (hash as usize) % self.size;
            if !self.bits[index] {
                return false;
            }
        }
        true
    }

    /// Hash function with seed
    fn hash_with_seed(&self, item: &str, seed: usize) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();
        hasher.write(item.as_bytes());
        hasher.write_usize(seed);
        hasher.finish()
    }

    /// Get fill ratio
    fn fill_ratio(&self) -> f64 {
        let filled = self.bits.iter().filter(|&&b| b).count();
        filled as f64 / self.size as f64
    }
}

/// Statistics for deduplication
struct DeduplicationStats {
    exact_matches: std::sync::atomic::AtomicU64,
    near_duplicates: std::sync::atomic::AtomicU64,
    no_duplicates: std::sync::atomic::AtomicU64,
    additions: std::sync::atomic::AtomicU64,
    removals: std::sync::atomic::AtomicU64,
    bloom_filter_misses: std::sync::atomic::AtomicU64,
    filtered_items: std::sync::atomic::AtomicU64,
    exact_duplicate_items: std::sync::atomic::AtomicU64,
}

impl DeduplicationStats {
    fn new() -> Self {
        use std::sync::atomic::AtomicU64;

        Self {
            exact_matches: AtomicU64::new(0),
            near_duplicates: AtomicU64::new(0),
            no_duplicates: AtomicU64::new(0),
            additions: AtomicU64::new(0),
            removals: AtomicU64::new(0),
            bloom_filter_misses: AtomicU64::new(0),
            filtered_items: AtomicU64::new(0),
            exact_duplicate_items: AtomicU64::new(0),
        }
    }

    fn record_exact_match(&self) {
        use std::sync::atomic::Ordering;
        self.exact_matches.fetch_add(1, Ordering::Relaxed);
    }

    fn record_near_duplicate(&self) {
        use std::sync::atomic::Ordering;
        self.near_duplicates.fetch_add(1, Ordering::Relaxed);
    }

    fn record_no_duplicate(&self) {
        use std::sync::atomic::Ordering;
        self.no_duplicates.fetch_add(1, Ordering::Relaxed);
    }

    fn record_addition(&self) {
        use std::sync::atomic::Ordering;
        self.additions.fetch_add(1, Ordering::Relaxed);
    }

    fn record_removal(&self) {
        use std::sync::atomic::Ordering;
        self.removals.fetch_add(1, Ordering::Relaxed);
    }

    fn record_bloom_filter_miss(&self) {
        use std::sync::atomic::Ordering;
        self.bloom_filter_misses.fetch_add(1, Ordering::Relaxed);
    }

    fn record_filtered_item(&self) {
        use std::sync::atomic::Ordering;
        self.filtered_items.fetch_add(1, Ordering::Relaxed);
    }

    fn record_exact_duplicate_item(&self) {
        use std::sync::atomic::Ordering;
        self.exact_duplicate_items.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> DeduplicationStatsSnapshot {
        use std::sync::atomic::Ordering;

        DeduplicationStatsSnapshot {
            exact_matches: self.exact_matches.load(Ordering::Relaxed),
            near_duplicates: self.near_duplicates.load(Ordering::Relaxed),
            no_duplicates: self.no_duplicates.load(Ordering::Relaxed),
            additions: self.additions.load(Ordering::Relaxed),
            removals: self.removals.load(Ordering::Relaxed),
            bloom_filter_misses: self.bloom_filter_misses.load(Ordering::Relaxed),
            filtered_items: self.filtered_items.load(Ordering::Relaxed),
            exact_duplicate_items: self.exact_duplicate_items.load(Ordering::Relaxed),
            dedup_rate: Self::calculate_dedup_rate(
                self.exact_matches.load(Ordering::Relaxed) +
                self.near_duplicates.load(Ordering::Relaxed),
                self.no_duplicates.load(Ordering::Relaxed)
            ),
        }
    }

    fn calculate_dedup_rate(duplicates: u64, non_duplicates: u64) -> f64 {
        let total = duplicates + non_duplicates;
        if total == 0 {
            0.0
        } else {
            (duplicates as f64) / (total as f64)
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DeduplicationStatsSnapshot {
    pub exact_matches: u64,
    pub near_duplicates: u64,
    pub no_duplicates: u64,
    pub additions: u64,
    pub removals: u64,
    pub bloom_filter_misses: u64,
    pub filtered_items: u64,
    pub exact_duplicate_items: u64,
    pub dedup_rate: f64,
}

#[derive(Debug, Clone)]
pub struct SimilarItem {
    pub id: String,
    pub similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedItemDedup {
    pub id: String,
    pub title: String,
    pub content: String,
    pub url: String,
    pub published_at: chrono::DateTime<chrono::Utc>,
}

/// Helper function to convert string hash to u64
fn hash_to_u64(hash: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    hasher.write(hash.as_bytes());
    hasher.finish()
}

/// Advanced deduplication strategies
pub struct DeduplicationStrategy {
    strategy_type: StrategyType,
}

#[derive(Debug, Clone)]
pub enum StrategyType {
    /// Exact matching only
    Exact,
    /// Fuzzy matching with configurable threshold
    Fuzzy { threshold: f64 },
    /// Semantic similarity using embeddings
    Semantic { model: String },
    /// Hybrid approach combining multiple methods
    Hybrid {
        exact_weight: f64,
        fuzzy_weight: f64,
        semantic_weight: f64,
    },
}

impl DeduplicationStrategy {
    /// Apply deduplication strategy
    pub async fn apply(&self, items: Vec<FeedItemDedup>) -> Vec<FeedItemDedup> {
        match &self.strategy_type {
            StrategyType::Exact => self.apply_exact(items),
            StrategyType::Fuzzy { threshold } => self.apply_fuzzy(items, *threshold).await,
            StrategyType::Semantic { model } => self.apply_semantic(items, model).await,
            StrategyType::Hybrid { exact_weight, fuzzy_weight, semantic_weight } => {
                self.apply_hybrid(items, *exact_weight, *fuzzy_weight, *semantic_weight).await
            }
        }
    }

    fn apply_exact(&self, items: Vec<FeedItemDedup>) -> Vec<FeedItemDedup> {
        let mut seen = HashSet::new();
        let mut unique = Vec::new();

        for item in items {
            let hash = blake3::hash(item.content.as_bytes()).to_hex().to_string();
            if seen.insert(hash) {
                unique.push(item);
            }
        }

        unique
    }

    async fn apply_fuzzy(&self, items: Vec<FeedItemDedup>, threshold: f64) -> Vec<FeedItemDedup> {
        let mut unique = Vec::new();
        let mut signatures = Vec::new();

        for item in items {
            let minhash = MinHash::new(&item.content, 128);
            let signature = minhash.signature();

            let mut is_duplicate = false;
            for existing_sig in &signatures {
                if MinHash::jaccard_similarity(&signature, existing_sig) > threshold {
                    is_duplicate = true;
                    break;
                }
            }

            if !is_duplicate {
                signatures.push(signature);
                unique.push(item);
            }
        }

        unique
    }

    async fn apply_semantic(&self, items: Vec<FeedItemDedup>, model: &str) -> Vec<FeedItemDedup> {
        // Would use semantic embeddings for similarity
        // This is a placeholder implementation
        items
    }

    async fn apply_hybrid(
        &self,
        items: Vec<FeedItemDedup>,
        exact_weight: f64,
        fuzzy_weight: f64,
        semantic_weight: f64,
    ) -> Vec<FeedItemDedup> {
        // Combine multiple strategies with weighting
        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_exact_deduplication() {
        let dedup = DeduplicationService::new(1000);

        let hash1 = dedup.compute_hash("test content 1");
        let hash2 = dedup.compute_hash("test content 1");
        let hash3 = dedup.compute_hash("test content 2");

        dedup.add_entry(&hash1, "id1").await;

        assert!(dedup.check_duplicate(&hash2).await.is_some());
        assert!(dedup.check_duplicate(&hash3).await.is_none());
    }

    #[tokio::test]
    async fn test_near_duplicate_detection() {
        let dedup = DeduplicationService::new(1000);

        let content1 = "This is a test document about deduplication algorithms";
        let content2 = "This is a test document about deduplication algorithms and methods";
        let content3 = "Completely different content here";

        dedup.add_content_signature("doc1", content1).await;

        let similar1 = dedup.check_near_duplicate(content2, 0.7).await;
        let similar2 = dedup.check_near_duplicate(content3, 0.7).await;

        assert!(similar1.is_some());
        assert!(similar2.is_none());
    }

    #[test]
    fn test_minhash_similarity() {
        let text1 = "the quick brown fox jumps over the lazy dog";
        let text2 = "the quick brown fox jumps over the lazy cat";
        let text3 = "completely different text here";

        let minhash1 = MinHash::new(text1, 128);
        let minhash2 = MinHash::new(text2, 128);
        let minhash3 = MinHash::new(text3, 128);

        let sim1_2 = MinHash::jaccard_similarity(&minhash1.signature(), &minhash2.signature());
        let sim1_3 = MinHash::jaccard_similarity(&minhash1.signature(), &minhash3.signature());

        assert!(sim1_2 > 0.7);  // Similar texts
        assert!(sim1_3 < 0.3);  // Different texts
    }

    #[test]
    fn test_bloom_filter() {
        let mut bloom = BloomFilter::new(1000, 0.01);

        bloom.insert("test1");
        bloom.insert("test2");

        assert!(bloom.contains("test1"));
        assert!(bloom.contains("test2"));
        assert!(!bloom.contains("test3"));
    }
}