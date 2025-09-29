use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    One,      // At least one replica
    Quorum,   // Majority of replicas
    All,      // All replicas
    Local,    // Local datacenter only
    Each,     // At least one in each datacenter
}

pub struct ConsistencyChecker {
    level: ConsistencyLevel,
}

impl ConsistencyChecker {
    pub fn new(level: ConsistencyLevel) -> Self {
        Self { level }
    }

    pub fn check_write_consistency(&self, successful_writes: usize, total_replicas: usize) -> bool {
        match self.level {
            ConsistencyLevel::One => successful_writes >= 1,
            ConsistencyLevel::Quorum => successful_writes > total_replicas / 2,
            ConsistencyLevel::All => successful_writes == total_replicas,
            ConsistencyLevel::Local => successful_writes >= 1, // Simplified
            ConsistencyLevel::Each => successful_writes >= 1, // Simplified
        }
    }

    pub fn check_read_consistency(&self, successful_reads: usize, total_replicas: usize) -> bool {
        match self.level {
            ConsistencyLevel::One => successful_reads >= 1,
            ConsistencyLevel::Quorum => successful_reads > total_replicas / 2,
            ConsistencyLevel::All => successful_reads == total_replicas,
            ConsistencyLevel::Local => successful_reads >= 1,
            ConsistencyLevel::Each => successful_reads >= 1,
        }
    }

    pub fn required_acks(&self, total_replicas: usize) -> usize {
        match self.level {
            ConsistencyLevel::One => 1,
            ConsistencyLevel::Quorum => (total_replicas / 2) + 1,
            ConsistencyLevel::All => total_replicas,
            ConsistencyLevel::Local => 1,
            ConsistencyLevel::Each => 1,
        }
    }
}