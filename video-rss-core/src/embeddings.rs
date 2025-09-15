use crate::{error::VideoRssError, Result};
use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::{Tokenizer, Encoding};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_id: String,
    pub model_type: EmbeddingModel,
    pub device: DeviceConfig,
    pub max_length: usize,
    pub batch_size: usize,
    pub normalize: bool,
    pub pooling: PoolingStrategy,
    pub cache_embeddings: bool,
    pub max_concurrent: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingModel {
    AllMiniLML6V2,      // 384 dims, fast
    AllMpnetBaseV2,     // 768 dims, balanced
    E5Small,            // 384 dims, multilingual
    E5Base,             // 768 dims, multilingual
    BGESmall,           // 384 dims, SOTA small
    BGEBase,            // 768 dims, SOTA base
    JinaEmbeddingsV2,   // 512 dims, 8K context
    NoMicUp,            // 768 dims, long context
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    Cpu,
    Cuda(usize),
    Metal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingStrategy {
    Mean,
    Max,
    CLS,
    MeanSqrt,
    WeightedMean,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            model_type: EmbeddingModel::AllMiniLML6V2,
            device: DeviceConfig::Cpu,
            max_length: 512,
            batch_size: 32,
            normalize: true,
            pooling: PoolingStrategy::Mean,
            cache_embeddings: true,
            max_concurrent: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResult {
    pub embedding: Vec<f32>,
    pub dimension: usize,
    pub tokens_processed: usize,
    pub processing_time_ms: u64,
}

pub struct EmbeddingGenerator {
    model: Arc<BertModel>,
    tokenizer: Arc<Tokenizer>,
    config: EmbeddingConfig,
    device: Device,
    embedding_cache: Arc<Mutex<lru::LruCache<String, Vec<f32>>>>,
    semaphore: Arc<Semaphore>,
}

impl EmbeddingGenerator {
    pub async fn new(config: EmbeddingConfig) -> Result<Self> {
        info!("Initializing embedding generator with model: {}", config.model_id);

        // Set up device
        let device = match &config.device {
            DeviceConfig::Cpu => Device::Cpu,
            DeviceConfig::Cuda(id) => Device::new_cuda(*id)
                .map_err(|e| VideoRssError::Config(format!("CUDA device error: {}", e)))?,
            DeviceConfig::Metal => Device::new_metal(0)
                .map_err(|e| VideoRssError::Config(format!("Metal device error: {}", e)))?,
        };

        // Download model from HuggingFace
        let api = Api::new()
            .map_err(|e| VideoRssError::Config(format!("Failed to create HF API: {}", e)))?;

        let repo = api.repo(Repo::model(config.model_id.clone()));

        // Download model files
        let model_file = repo
            .get("pytorch_model.bin")
            .await
            .or_else(|_| repo.get("model.safetensors").await)
            .map_err(|e| VideoRssError::Config(format!("Failed to download model: {}", e)))?;

        let config_file = repo
            .get("config.json")
            .await
            .map_err(|e| VideoRssError::Config(format!("Failed to download config: {}", e)))?;

        let tokenizer_file = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| VideoRssError::Config(format!("Failed to download tokenizer: {}", e)))?;

        // Load configuration
        let config_str = std::fs::read_to_string(&config_file)
            .map_err(|e| VideoRssError::Io(e))?;
        let bert_config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| VideoRssError::Config(format!("Failed to parse config: {}", e)))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)
                .map_err(|e| VideoRssError::Config(format!("Failed to load model: {}", e)))?
        };

        // Create BERT model
        let model = BertModel::load(vb, &bert_config)
            .map_err(|e| VideoRssError::Config(format!("Failed to create model: {}", e)))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| VideoRssError::Config(format!("Failed to load tokenizer: {}", e)))?;

        // Initialize cache
        let cache = lru::LruCache::new(std::num::NonZeroUsize::new(1000).unwrap());

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));

        info!("Embedding generator initialized successfully");

        Ok(Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            config,
            device,
            embedding_cache: Arc::new(Mutex::new(cache)),
            semaphore,
        })
    }

    pub async fn generate(&self, text: &str) -> Result<EmbeddingResult> {
        // Check cache first
        if self.config.cache_embeddings {
            let mut cache = self.embedding_cache.lock().await;
            if let Some(cached) = cache.get(text) {
                debug!("Embedding cache hit for text");
                return Ok(EmbeddingResult {
                    embedding: cached.clone(),
                    dimension: cached.len(),
                    tokens_processed: 0,
                    processing_time_ms: 0,
                });
            }
        }

        let _permit = self.semaphore.acquire().await
            .map_err(|e| VideoRssError::Unknown(format!("Semaphore error: {}", e)))?;

        let start_time = std::time::Instant::now();

        // Tokenize text
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| VideoRssError::Unknown(format!("Tokenization error: {}", e)))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let tokens_processed = input_ids.len();

        // Truncate if necessary
        let (input_ids, attention_mask) = if input_ids.len() > self.config.max_length {
            (
                &input_ids[..self.config.max_length],
                &attention_mask[..self.config.max_length],
            )
        } else {
            (input_ids, attention_mask)
        };

        // Convert to tensors
        let input_ids_tensor = Tensor::new(input_ids, &self.device)
            .map_err(|e| VideoRssError::Unknown(format!("Tensor creation error: {}", e)))?
            .unsqueeze(0)?;  // Add batch dimension

        let attention_mask_tensor = Tensor::new(attention_mask, &self.device)
            .map_err(|e| VideoRssError::Unknown(format!("Tensor creation error: {}", e)))?
            .unsqueeze(0)?;

        // Forward pass
        let output = self.model.forward(&input_ids_tensor, &attention_mask_tensor)
            .map_err(|e| VideoRssError::Unknown(format!("Model forward error: {}", e)))?;

        // Apply pooling strategy
        let pooled = self.apply_pooling(&output, &attention_mask_tensor)?;

        // Convert to Vec<f32>
        let embedding: Vec<f32> = pooled.squeeze(0)?
            .to_vec1()
            .map_err(|e| VideoRssError::Unknown(format!("Tensor conversion error: {}", e)))?;

        // Normalize if configured
        let embedding = if self.config.normalize {
            self.normalize_embedding(embedding)
        } else {
            embedding
        };

        // Cache the result
        if self.config.cache_embeddings {
            let mut cache = self.embedding_cache.lock().await;
            cache.put(text.to_string(), embedding.clone());
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(EmbeddingResult {
            dimension: embedding.len(),
            embedding,
            tokens_processed,
            processing_time_ms,
        })
    }

    fn apply_pooling(
        &self,
        output: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        match self.config.pooling {
            PoolingStrategy::Mean => {
                // Mean pooling with attention mask
                let sum = (output * attention_mask.unsqueeze(-1)?)?
                    .sum(1)?;
                let count = attention_mask.sum(1)?.unsqueeze(-1)?;
                Ok((sum / count)?)
            },
            PoolingStrategy::Max => {
                // Max pooling
                Ok(output.max(1)?)
            },
            PoolingStrategy::CLS => {
                // Use CLS token (first token)
                Ok(output.i((.., 0, ..))?)
            },
            PoolingStrategy::MeanSqrt => {
                // Mean pooling with sqrt length normalization
                let sum = (output * attention_mask.unsqueeze(-1)?)?
                    .sum(1)?;
                let count = attention_mask.sum(1)?.unsqueeze(-1)?;
                let mean = sum / count;
                Ok((mean * count.sqrt())?)
            },
            PoolingStrategy::WeightedMean => {
                // Weighted mean with position-based weights
                let seq_len = output.dims()[1];
                let weights = Tensor::arange(0u32, seq_len as u32, &self.device)?
                    .to_dtype(DType::F32)?;
                let weights = (weights / seq_len as f32)?;

                let weighted = output * weights.unsqueeze(0)?.unsqueeze(-1)?;
                let sum = (weighted * attention_mask.unsqueeze(-1)?)?
                    .sum(1)?;
                let count = attention_mask.sum(1)?.unsqueeze(-1)?;
                Ok((sum / count)?)
            },
        }
    }

    fn normalize_embedding(&self, mut embedding: Vec<f32>) -> Vec<f32> {
        let norm = embedding.iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    pub async fn batch_generate(&self, texts: Vec<String>) -> Result<Vec<EmbeddingResult>> {
        let mut results = Vec::new();

        // Process in batches
        for chunk in texts.chunks(self.config.batch_size) {
            let batch_results = self.generate_batch(chunk).await?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    async fn generate_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingResult>> {
        let start_time = std::time::Instant::now();

        // Tokenize all texts
        let mut all_input_ids = Vec::new();
        let mut all_attention_masks = Vec::new();
        let mut max_len = 0;

        for text in texts {
            let encoding = self.tokenizer.encode(text, true)
                .map_err(|e| VideoRssError::Unknown(format!("Tokenization error: {}", e)))?;

            let input_ids = encoding.get_ids().to_vec();
            let attention_mask = encoding.get_attention_mask().to_vec();

            max_len = max_len.max(input_ids.len().min(self.config.max_length));

            all_input_ids.push(input_ids);
            all_attention_masks.push(attention_mask);
        }

        // Pad to same length
        for i in 0..texts.len() {
            all_input_ids[i].truncate(max_len);
            all_attention_masks[i].truncate(max_len);

            while all_input_ids[i].len() < max_len {
                all_input_ids[i].push(0);  // Padding token
                all_attention_masks[i].push(0);
            }
        }

        // Convert to tensors
        let input_ids_flat: Vec<u32> = all_input_ids.concat();
        let attention_mask_flat: Vec<u32> = all_attention_masks.concat();

        let input_ids_tensor = Tensor::from_vec(
            input_ids_flat,
            (texts.len(), max_len),
            &self.device,
        )?;

        let attention_mask_tensor = Tensor::from_vec(
            attention_mask_flat,
            (texts.len(), max_len),
            &self.device,
        )?;

        // Forward pass
        let output = self.model.forward(&input_ids_tensor, &attention_mask_tensor)
            .map_err(|e| VideoRssError::Unknown(format!("Model forward error: {}", e)))?;

        // Apply pooling and convert to results
        let mut results = Vec::new();
        for i in 0..texts.len() {
            let single_output = output.i((i, .., ..))?;
            let single_mask = attention_mask_tensor.i((i, ..))?;

            let pooled = self.apply_pooling(&single_output.unsqueeze(0)?, &single_mask.unsqueeze(0)?)?;
            let embedding: Vec<f32> = pooled.squeeze(0)?
                .to_vec1()
                .map_err(|e| VideoRssError::Unknown(format!("Tensor conversion error: {}", e)))?;

            let embedding = if self.config.normalize {
                self.normalize_embedding(embedding)
            } else {
                embedding
            };

            results.push(EmbeddingResult {
                dimension: embedding.len(),
                embedding,
                tokens_processed: max_len,
                processing_time_ms: 0,  // Will be set below
            });
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        for result in &mut results {
            result.processing_time_ms = processing_time_ms / texts.len() as u64;
        }

        Ok(results)
    }

    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter())
            .map(|(x, y)| x * y)
            .sum();

        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }

        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}