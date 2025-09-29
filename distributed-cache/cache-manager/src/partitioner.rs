use consistent_hash::ConsistentHash;
use rendezvous_hash::RendezvousNodes;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use xxhash_rust::xxh3::Xxh3;

#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    Hash,
    ConsistentHash,
    RendezvousHash,
    Range,
    Custom,
}

pub struct Partitioner {
    pub strategy: PartitionStrategy,
    partition_count: usize,
    consistent_hash: Option<ConsistentHash<usize>>,
    rendezvous_nodes: Option<RendezvousNodes<usize>>,
    range_map: Option<RangePartitioner>,
}

impl Partitioner {
    pub fn new(strategy: PartitionStrategy, partition_count: usize) -> Self {
        let mut partitioner = Self {
            strategy: strategy.clone(),
            partition_count,
            consistent_hash: None,
            rendezvous_nodes: None,
            range_map: None,
        };

        match strategy {
            PartitionStrategy::ConsistentHash => {
                let mut ch = ConsistentHash::new();
                for i in 0..partition_count {
                    // Add virtual nodes for better distribution
                    for j in 0..150 {
                        ch.add(format!("node_{}_{}", i, j), i);
                    }
                }
                partitioner.consistent_hash = Some(ch);
            }
            PartitionStrategy::RendezvousHash => {
                let mut nodes = RendezvousNodes::default();
                for i in 0..partition_count {
                    nodes.insert(i);
                }
                partitioner.rendezvous_nodes = Some(nodes);
            }
            PartitionStrategy::Range => {
                partitioner.range_map = Some(RangePartitioner::new(partition_count));
            }
            _ => {}
        }

        partitioner
    }

    pub fn get_partition(&self, key: &str) -> usize {
        match self.strategy {
            PartitionStrategy::Hash => self.hash_partition(key),
            PartitionStrategy::ConsistentHash => self.consistent_hash_partition(key),
            PartitionStrategy::RendezvousHash => self.rendezvous_hash_partition(key),
            PartitionStrategy::Range => self.range_partition(key),
            PartitionStrategy::Custom => self.custom_partition(key),
        }
    }

    fn hash_partition(&self, key: &str) -> usize {
        let mut hasher = Xxh3::new();
        hasher.update(key.as_bytes());
        let hash = hasher.digest();
        (hash % self.partition_count as u64) as usize
    }

    fn consistent_hash_partition(&self, key: &str) -> usize {
        if let Some(ref ch) = self.consistent_hash {
            *ch.get(key).unwrap_or(&0)
        } else {
            self.hash_partition(key)
        }
    }

    fn rendezvous_hash_partition(&self, key: &str) -> usize {
        if let Some(ref nodes) = self.rendezvous_nodes {
            nodes.calc_candidates(key.as_bytes())
                .first()
                .map(|n| n.node)
                .copied()
                .unwrap_or(0)
        } else {
            self.hash_partition(key)
        }
    }

    fn range_partition(&self, key: &str) -> usize {
        if let Some(ref rm) = self.range_map {
            rm.get_partition(key)
        } else {
            self.hash_partition(key)
        }
    }

    fn custom_partition(&self, key: &str) -> usize {
        // Custom partitioning logic based on key patterns
        if key.starts_with("video:") {
            // Videos go to partitions 0-3
            self.hash_partition(key) % 4
        } else if key.starts_with("transcript:") {
            // Transcripts go to partitions 4-7
            4 + (self.hash_partition(key) % 4)
        } else if key.starts_with("summary:") {
            // Summaries go to partitions 8-11
            8 + (self.hash_partition(key) % 4)
        } else {
            // Others distributed across all partitions
            self.hash_partition(key)
        }
    }

    pub fn get_partition_count(&self) -> usize {
        self.partition_count
    }

    pub fn list_all_shards(&self) -> Vec<ShardInfo> {
        (0..self.partition_count)
            .map(|i| ShardInfo {
                id: i,
                node_count: 3, // Assuming 3 replicas per shard
                key_range: self.get_shard_key_range(i),
                status: "active".to_string(),
            })
            .collect()
    }

    fn get_shard_key_range(&self, shard_id: usize) -> String {
        match self.strategy {
            PartitionStrategy::Range => {
                if let Some(ref rm) = self.range_map {
                    rm.get_range_for_shard(shard_id)
                } else {
                    format!("shard_{}", shard_id)
                }
            }
            _ => format!("hash_shard_{}", shard_id),
        }
    }

    pub fn rebalance(&mut self, new_partition_count: usize) {
        // Implement rebalancing logic
        self.partition_count = new_partition_count;

        match self.strategy {
            PartitionStrategy::ConsistentHash => {
                let mut ch = ConsistentHash::new();
                for i in 0..new_partition_count {
                    for j in 0..150 {
                        ch.add(format!("node_{}_{}", i, j), i);
                    }
                }
                self.consistent_hash = Some(ch);
            }
            PartitionStrategy::RendezvousHash => {
                let mut nodes = RendezvousNodes::default();
                for i in 0..new_partition_count {
                    nodes.insert(i);
                }
                self.rendezvous_nodes = Some(nodes);
            }
            _ => {}
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ShardInfo {
    pub id: usize,
    pub node_count: usize,
    pub key_range: String,
    pub status: String,
}

struct RangePartitioner {
    ranges: Vec<(String, String, usize)>, // (start, end, partition)
}

impl RangePartitioner {
    fn new(partition_count: usize) -> Self {
        let mut ranges = Vec::new();
        let chars = "0123456789abcdefghijklmnopqrstuvwxyz";
        let step = chars.len() / partition_count;

        for i in 0..partition_count {
            let start_idx = i * step;
            let end_idx = if i == partition_count - 1 {
                chars.len()
            } else {
                (i + 1) * step
            };

            let start = chars.chars().nth(start_idx).unwrap_or('0').to_string();
            let end = chars.chars().nth(end_idx.saturating_sub(1)).unwrap_or('z').to_string();

            ranges.push((start, end, i));
        }

        Self { ranges }
    }

    fn get_partition(&self, key: &str) -> usize {
        let first_char = key.chars().next().unwrap_or('0').to_lowercase().to_string();

        for (start, end, partition) in &self.ranges {
            if first_char >= *start && first_char <= *end {
                return *partition;
            }
        }

        0 // Default partition
    }

    fn get_range_for_shard(&self, shard_id: usize) -> String {
        for (start, end, partition) in &self.ranges {
            if *partition == shard_id {
                return format!("[{}-{}]", start, end);
            }
        }
        "unknown".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_partition() {
        let partitioner = Partitioner::new(PartitionStrategy::Hash, 10);

        let partition1 = partitioner.get_partition("key1");
        let partition2 = partitioner.get_partition("key1");
        assert_eq!(partition1, partition2); // Same key should map to same partition

        let partition3 = partitioner.get_partition("key2");
        // Different keys might map to same partition, but that's ok
        assert!(partition3 < 10);
    }

    #[test]
    fn test_custom_partition() {
        let partitioner = Partitioner::new(PartitionStrategy::Custom, 16);

        let video_partition = partitioner.get_partition("video:123");
        assert!(video_partition < 4);

        let transcript_partition = partitioner.get_partition("transcript:123");
        assert!(transcript_partition >= 4 && transcript_partition < 8);

        let summary_partition = partitioner.get_partition("summary:123");
        assert!(summary_partition >= 8 && summary_partition < 12);
    }

    #[test]
    fn test_range_partition() {
        let partitioner = Partitioner::new(PartitionStrategy::Range, 4);

        let partition_a = partitioner.get_partition("apple");
        let partition_z = partitioner.get_partition("zebra");

        // Keys starting with 'a' should be in lower partitions
        // Keys starting with 'z' should be in higher partitions
        assert!(partition_a <= partition_z);
    }
}