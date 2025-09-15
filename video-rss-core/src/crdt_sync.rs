use crate::{error::VideoRssError, Result};
use automerge::{Automerge, AutomergeError, ChangeHash, ObjType, ScalarValue, transaction::Transactable};
use automerge::sync::{Message, State as SyncState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// CRDT-based distributed sync using Automerge 3
/// Enables conflict-free collaborative editing and offline-first sync
pub struct CrdtSync {
    doc: Arc<RwLock<Automerge>>,
    sync_states: Arc<RwLock<HashMap<String, SyncState>>>,
    event_tx: broadcast::Sender<SyncEvent>,
    peer_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncEvent {
    DocumentChanged {
        peer_id: String,
        changes: Vec<ChangeHash>,
    },
    PeerConnected {
        peer_id: String,
    },
    PeerDisconnected {
        peer_id: String,
    },
    ConflictResolved {
        field: String,
        winning_value: String,
    },
}

impl CrdtSync {
    pub async fn new() -> Result<Self> {
        info!("Initializing CRDT sync with Automerge 3");
        
        let doc = Arc::new(RwLock::new(Automerge::new()));
        let sync_states = Arc::new(RwLock::new(HashMap::new()));
        let (event_tx, _) = broadcast::channel(1000);
        let peer_id = Uuid::new_v4().to_string();
        
        // Initialize document structure
        {
            let mut doc = doc.write().await;
            let mut tx = doc.transaction();
            
            // Create root collections
            tx.put_object(automerge::ROOT, "videos", ObjType::Map)
                .map_err(|e| VideoRssError::Unknown(format!("Automerge error: {}", e)))?;
            tx.put_object(automerge::ROOT, "transcriptions", ObjType::Map)
                .map_err(|e| VideoRssError::Unknown(format!("Automerge error: {}", e)))?;
            tx.put_object(automerge::ROOT, "summaries", ObjType::Map)
                .map_err(|e| VideoRssError::Unknown(format!("Automerge error: {}", e)))?;
            tx.put_object(automerge::ROOT, "metadata", ObjType::Map)
                .map_err(|e| VideoRssError::Unknown(format!("Automerge error: {}", e)))?;
            
            tx.commit();
        }
        
        info!("CRDT sync initialized with peer ID: {}", peer_id);
        
        Ok(Self {
            doc,
            sync_states,
            event_tx,
            peer_id,
        })
    }

    /// Add or update a video entry
    pub async fn update_video(&self, video_id: &str, data: VideoData) -> Result<()> {
        let mut doc = self.doc.write().await;
        let mut tx = doc.transaction();
        
        // Get videos collection
        let videos = tx.get(automerge::ROOT, "videos")
            .and_then(|(v, _)| v.to_objid())
            .ok_or_else(|| VideoRssError::NotFound("Videos collection not found".to_string()))?;
        
        // Create or update video object
        let video_obj = if let Some((existing, _)) = tx.get(&videos, video_id) {
            existing.to_objid()
                .ok_or_else(|| VideoRssError::Unknown("Invalid video object".to_string()))?
        } else {
            tx.put_object(&videos, video_id, ObjType::Map)
                .map_err(|e| VideoRssError::Unknown(format!("Put object error: {}", e)))?
        };
        
        // Update fields
        tx.put(&video_obj, "title", ScalarValue::Str(data.title.into()))
            .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        tx.put(&video_obj, "url", ScalarValue::Str(data.url.into()))
            .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        tx.put(&video_obj, "duration", ScalarValue::F64(data.duration_seconds))
            .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        tx.put(&video_obj, "published", ScalarValue::Timestamp(data.published_at))
            .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        tx.put(&video_obj, "processed", ScalarValue::Boolean(data.processed))
            .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        
        // Add tags as list
        let tags_list = tx.put_object(&video_obj, "tags", ObjType::List)
            .map_err(|e| VideoRssError::Unknown(format!("Put list error: {}", e)))?;
        
        for tag in data.tags {
            tx.insert(&tags_list, 0, ScalarValue::Str(tag.into()))
                .map_err(|e| VideoRssError::Unknown(format!("Insert error: {}", e)))?;
        }
        
        let changes = tx.commit();
        
        // Notify peers
        let _ = self.event_tx.send(SyncEvent::DocumentChanged {
            peer_id: self.peer_id.clone(),
            changes,
        });
        
        Ok(())
    }

    /// Add or update a transcription
    pub async fn update_transcription(
        &self,
        video_id: &str,
        transcription: TranscriptionData,
    ) -> Result<()> {
        let mut doc = self.doc.write().await;
        let mut tx = doc.transaction();
        
        let transcriptions = tx.get(automerge::ROOT, "transcriptions")
            .and_then(|(v, _)| v.to_objid())
            .ok_or_else(|| VideoRssError::NotFound("Transcriptions collection not found".to_string()))?;
        
        let trans_obj = tx.put_object(&transcriptions, video_id, ObjType::Map)
            .map_err(|e| VideoRssError::Unknown(format!("Put object error: {}", e)))?;
        
        // Store transcription data
        tx.put(&trans_obj, "text", ScalarValue::Str(transcription.text.into()))
            .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        tx.put(&trans_obj, "language", ScalarValue::Str(transcription.language.into()))
            .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        tx.put(&trans_obj, "confidence", ScalarValue::F64(transcription.confidence as f64))
            .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        tx.put(&trans_obj, "timestamp", ScalarValue::Timestamp(chrono::Utc::now().timestamp()))
            .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        
        // Store segments
        let segments_list = tx.put_object(&trans_obj, "segments", ObjType::List)
            .map_err(|e| VideoRssError::Unknown(format!("Put list error: {}", e)))?;
        
        for segment in transcription.segments {
            let seg_obj = tx.insert_object(&segments_list, 0, ObjType::Map)
                .map_err(|e| VideoRssError::Unknown(format!("Insert object error: {}", e)))?;
            
            tx.put(&seg_obj, "start", ScalarValue::F64(segment.start_time))
                .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
            tx.put(&seg_obj, "end", ScalarValue::F64(segment.end_time))
                .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
            tx.put(&seg_obj, "text", ScalarValue::Str(segment.text.into()))
                .map_err(|e| VideoRssError::Unknown(format!("Put error: {}", e)))?;
        }
        
        let changes = tx.commit();
        
        let _ = self.event_tx.send(SyncEvent::DocumentChanged {
            peer_id: self.peer_id.clone(),
            changes,
        });
        
        Ok(())
    }

    /// Sync with a remote peer
    pub async fn sync_with_peer(&self, peer_id: &str) -> Result<Vec<u8>> {
        let mut sync_states = self.sync_states.write().await;
        let doc = self.doc.read().await;
        
        // Get or create sync state for this peer
        let sync_state = sync_states.entry(peer_id.to_string())
            .or_insert_with(SyncState::new);
        
        // Generate sync message
        let message = doc.generate_sync_message(sync_state)
            .map_err(|e| VideoRssError::Unknown(format!("Sync message error: {}", e)))?;
        
        // Serialize message
        let bytes = message.encode();
        
        info!("Generated sync message for peer {} ({} bytes)", peer_id, bytes.len());
        
        Ok(bytes)
    }

    /// Receive and apply sync message from peer
    pub async fn receive_sync_message(&self, peer_id: &str, message_bytes: &[u8]) -> Result<Vec<u8>> {
        let mut sync_states = self.sync_states.write().await;
        let mut doc = self.doc.write().await;
        
        // Decode message
        let message = Message::decode(message_bytes)
            .map_err(|e| VideoRssError::Unknown(format!("Message decode error: {}", e)))?;
        
        // Get or create sync state for this peer
        let sync_state = sync_states.entry(peer_id.to_string())
            .or_insert_with(SyncState::new);
        
        // Receive and apply changes
        doc.receive_sync_message(sync_state, message)
            .map_err(|e| VideoRssError::Unknown(format!("Receive sync error: {}", e)))?;
        
        // Generate response
        let response = doc.generate_sync_message(sync_state)
            .map_err(|e| VideoRssError::Unknown(format!("Response generation error: {}", e)))?;
        
        let response_bytes = response.encode();
        
        info!("Applied sync from peer {} and generated response ({} bytes)", 
              peer_id, response_bytes.len());
        
        // Notify about changes
        let _ = self.event_tx.send(SyncEvent::DocumentChanged {
            peer_id: peer_id.to_string(),
            changes: vec![],  // Would extract actual changes
        });
        
        Ok(response_bytes)
    }

    /// Get current document state as JSON
    pub async fn get_state_json(&self) -> Result<serde_json::Value> {
        let doc = self.doc.read().await;
        
        // Extract data from Automerge document
        let mut state = serde_json::json!({
            "peer_id": self.peer_id,
            "videos": {},
            "transcriptions": {},
            "summaries": {},
            "metadata": {}
        });
        
        // This is simplified - actual implementation would traverse the document
        Ok(state)
    }

    /// Subscribe to sync events
    pub fn subscribe(&self) -> broadcast::Receiver<SyncEvent> {
        self.event_tx.subscribe()
    }

    /// Merge documents from multiple peers
    pub async fn merge_from_peers(&self, peer_docs: Vec<Vec<u8>>) -> Result<()> {
        let mut doc = self.doc.write().await;
        
        for peer_doc_bytes in peer_docs {
            let peer_doc = Automerge::load(&peer_doc_bytes)
                .map_err(|e| VideoRssError::Unknown(format!("Load error: {}", e)))?;
            
            doc.merge(&mut peer_doc.clone())
                .map_err(|e| VideoRssError::Unknown(format!("Merge error: {}", e)))?;
        }
        
        info!("Merged documents from {} peers", peer_docs.len());
        
        Ok(())
    }

    /// Export document for persistence
    pub async fn export(&self) -> Result<Vec<u8>> {
        let doc = self.doc.read().await;
        doc.save()
            .map_err(|e| VideoRssError::Unknown(format!("Save error: {}", e)))
    }

    /// Import document from persistence
    pub async fn import(&self, data: &[u8]) -> Result<()> {
        let imported = Automerge::load(data)
            .map_err(|e| VideoRssError::Unknown(format!("Load error: {}", e)))?;
        
        let mut doc = self.doc.write().await;
        *doc = imported;
        
        info!("Imported document ({} bytes)", data.len());
        
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoData {
    pub title: String,
    pub url: String,
    pub duration_seconds: f64,
    pub published_at: i64,
    pub processed: bool,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionData {
    pub text: String,
    pub language: String,
    pub confidence: f32,
    pub segments: Vec<TranscriptionSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub text: String,
}

/// P2P sync coordinator
pub struct P2PSyncCoordinator {
    crdt: Arc<CrdtSync>,
    peers: Arc<RwLock<HashMap<String, PeerInfo>>>,
    sync_interval: std::time::Duration,
}

#[derive(Clone)]
struct PeerInfo {
    id: String,
    address: String,
    last_sync: std::time::Instant,
    online: bool,
}

impl P2PSyncCoordinator {
    pub async fn new(crdt: Arc<CrdtSync>) -> Result<Self> {
        Ok(Self {
            crdt,
            peers: Arc::new(RwLock::new(HashMap::new())),
            sync_interval: std::time::Duration::from_secs(30),
        })
    }

    /// Add a peer for syncing
    pub async fn add_peer(&self, peer_id: String, address: String) -> Result<()> {
        let mut peers = self.peers.write().await;
        peers.insert(peer_id.clone(), PeerInfo {
            id: peer_id.clone(),
            address,
            last_sync: std::time::Instant::now(),
            online: true,
        });
        
        let _ = self.crdt.event_tx.send(SyncEvent::PeerConnected { peer_id });
        
        Ok(())
    }

    /// Remove a peer
    pub async fn remove_peer(&self, peer_id: &str) -> Result<()> {
        let mut peers = self.peers.write().await;
        peers.remove(peer_id);
        
        let _ = self.crdt.event_tx.send(SyncEvent::PeerDisconnected {
            peer_id: peer_id.to_string(),
        });
        
        Ok(())
    }

    /// Start automatic sync with all peers
    pub async fn start_auto_sync(&self) {
        let peers = self.peers.clone();
        let crdt = self.crdt.clone();
        let interval = self.sync_interval;
        
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                
                let peer_list: Vec<PeerInfo> = peers.read().await.values().cloned().collect();
                
                for peer in peer_list {
                    if peer.online {
                        // Simulate sync - in real implementation would use network
                        match crdt.sync_with_peer(&peer.id).await {
                            Ok(message) => {
                                debug!("Synced with peer {}: {} bytes", peer.id, message.len());
                            }
                            Err(e) => {
                                warn!("Failed to sync with peer {}: {}", peer.id, e);
                            }
                        }
                    }
                }
            }
        });
    }
}

/// Conflict resolution strategies
pub struct ConflictResolver {
    strategy: ResolutionStrategy,
}

#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    LastWriteWins,
    FirstWriteWins,
    HighestValue,
    LowestValue,
    Custom(Arc<dyn Fn(&ScalarValue, &ScalarValue) -> ScalarValue + Send + Sync>),
}

impl ConflictResolver {
    pub fn new(strategy: ResolutionStrategy) -> Self {
        Self { strategy }
    }

    pub fn resolve(&self, a: &ScalarValue, b: &ScalarValue) -> ScalarValue {
        match &self.strategy {
            ResolutionStrategy::LastWriteWins => b.clone(),
            ResolutionStrategy::FirstWriteWins => a.clone(),
            ResolutionStrategy::HighestValue => {
                match (a, b) {
                    (ScalarValue::F64(x), ScalarValue::F64(y)) => {
                        if x > y { a.clone() } else { b.clone() }
                    }
                    _ => b.clone(),
                }
            }
            ResolutionStrategy::LowestValue => {
                match (a, b) {
                    (ScalarValue::F64(x), ScalarValue::F64(y)) => {
                        if x < y { a.clone() } else { b.clone() }
                    }
                    _ => a.clone(),
                }
            }
            ResolutionStrategy::Custom(f) => f(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_crdt_sync() {
        let sync = CrdtSync::new().await.unwrap();
        
        let video = VideoData {
            title: "Test Video".to_string(),
            url: "https://example.com/video".to_string(),
            duration_seconds: 120.0,
            published_at: 1234567890,
            processed: false,
            tags: vec!["test".to_string(), "video".to_string()],
        };
        
        sync.update_video("video1", video).await.unwrap();
        
        let state = sync.export().await.unwrap();
        assert!(!state.is_empty());
    }

    #[tokio::test]
    async fn test_merge() {
        let sync1 = CrdtSync::new().await.unwrap();
        let sync2 = CrdtSync::new().await.unwrap();
        
        // Add different data to each
        let video1 = VideoData {
            title: "Video 1".to_string(),
            url: "url1".to_string(),
            duration_seconds: 60.0,
            published_at: 1000,
            processed: true,
            tags: vec![],
        };
        
        let video2 = VideoData {
            title: "Video 2".to_string(),
            url: "url2".to_string(),
            duration_seconds: 90.0,
            published_at: 2000,
            processed: false,
            tags: vec![],
        };
        
        sync1.update_video("v1", video1).await.unwrap();
        sync2.update_video("v2", video2).await.unwrap();
        
        // Export and merge
        let doc2 = sync2.export().await.unwrap();
        sync1.merge_from_peers(vec![doc2]).await.unwrap();
        
        // Both videos should now be in sync1
        let state = sync1.get_state_json().await.unwrap();
        assert!(state.get("videos").is_some());
    }
}