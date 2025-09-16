use crate::{error::VideoRssError, Result};
use arrow::array::{Float32Array, StringArray, Int64Array, StructArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use lancedb::{Connection, Table};
use lance::dataset::{Dataset, WriteParams};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tantivy::{
    collector::TopDocs,
    query::QueryParser,
    schema::*,
    Index, IndexReader, IndexWriter, ReloadPolicy,
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDBConfig {
    pub data_path: PathBuf,
    pub index_path: PathBuf,
    pub embedding_dim: usize,
    pub max_results: usize,
    pub use_gpu: bool,
    pub cache_size_mb: usize,
}

impl Default for VectorDBConfig {
    fn default() -> Self {
        Self {
            data_path: PathBuf::from("./data/lancedb"),
            index_path: PathBuf::from("./data/tantivy"),
            embedding_dim: 384,  // For all-MiniLM-L6-v2
            max_results: 100,
            use_gpu: false,
            cache_size_mb: 512,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoDocument {
    pub id: String,
    pub title: String,
    pub description: String,
    pub transcript: String,
    pub embedding: Vec<f32>,
    pub platform: String,
    pub author: String,
    pub url: String,
    pub upload_timestamp: i64,
    pub view_count: i64,
    pub duration_seconds: i64,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub document: VideoDocument,
    pub score: f32,
    pub distance: f32,
    pub highlights: Vec<String>,
}

pub struct VectorDatabase {
    lance_conn: Arc<Connection>,
    tantivy_index: Arc<Index>,
    index_writer: Arc<RwLock<IndexWriter>>,
    index_reader: IndexReader,
    config: VectorDBConfig,
    schema: Arc<Schema>,
    tantivy_schema: tantivy::schema::Schema,
}

impl VectorDatabase {
    pub async fn new(config: VectorDBConfig) -> Result<Self> {
        info!("Initializing vector database at {:?}", config.data_path);

        // Create directories
        std::fs::create_dir_all(&config.data_path)
            .map_err(|e| VideoRssError::Io(e))?;
        std::fs::create_dir_all(&config.index_path)
            .map_err(|e| VideoRssError::Io(e))?;

        // Initialize LanceDB connection
        let lance_conn = Connection::open(&config.data_path)
            .await
            .map_err(|e| VideoRssError::Config(format!("LanceDB connection error: {}", e)))?;

        // Create Arrow schema for video documents
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, false),
            Field::new("description", DataType::Utf8, true),
            Field::new("transcript", DataType::Utf8, true),
            Field::new(
                "embedding",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    config.embedding_dim as i32,
                ),
                false,
            ),
            Field::new("platform", DataType::Utf8, false),
            Field::new("author", DataType::Utf8, false),
            Field::new("url", DataType::Utf8, false),
            Field::new("upload_timestamp", DataType::Int64, false),
            Field::new("view_count", DataType::Int64, false),
            Field::new("duration_seconds", DataType::Int64, false),
            Field::new("tags", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), true),
        ]));

        // Initialize Tantivy for full-text search
        let mut tantivy_schema_builder = Schema::builder();

        tantivy_schema_builder.add_text_field("id", STRING | STORED);
        tantivy_schema_builder.add_text_field("title", TEXT | STORED);
        tantivy_schema_builder.add_text_field("description", TEXT | STORED);
        tantivy_schema_builder.add_text_field("transcript", TEXT | STORED);
        tantivy_schema_builder.add_text_field("platform", STRING | STORED);
        tantivy_schema_builder.add_text_field("author", TEXT | STORED);
        tantivy_schema_builder.add_text_field("url", STRING | STORED);
        tantivy_schema_builder.add_i64_field("upload_timestamp", INDEXED | STORED);
        tantivy_schema_builder.add_i64_field("view_count", INDEXED | STORED);
        tantivy_schema_builder.add_i64_field("duration_seconds", INDEXED | STORED);
        tantivy_schema_builder.add_text_field("tags", TEXT | STORED);

        let tantivy_schema = tantivy_schema_builder.build();

        // Create or open Tantivy index
        let tantivy_index = if config.index_path.exists() {
            Index::open_in_dir(&config.index_path)
                .map_err(|e| VideoRssError::Config(format!("Tantivy index open error: {}", e)))?
        } else {
            Index::create_in_dir(&config.index_path, tantivy_schema.clone())
                .map_err(|e| VideoRssError::Config(format!("Tantivy index create error: {}", e)))?
        };

        // Create index writer with configured memory budget
        let index_writer = tantivy_index
            .writer(config.cache_size_mb * 1_000_000)
            .map_err(|e| VideoRssError::Config(format!("Tantivy writer error: {}", e)))?;

        // Create index reader
        let index_reader = tantivy_index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()
            .map_err(|e| VideoRssError::Config(format!("Tantivy reader error: {}", e)))?;

        info!("Vector database initialized successfully");

        Ok(Self {
            lance_conn: Arc::new(lance_conn),
            tantivy_index: Arc::new(tantivy_index),
            index_writer: Arc::new(RwLock::new(index_writer)),
            index_reader,
            config,
            schema,
            tantivy_schema,
        })
    }

    pub async fn insert_document(&self, doc: VideoDocument) -> Result<()> {
        debug!("Inserting document: {}", doc.id);

        // Insert into LanceDB for vector search
        self.insert_to_lancedb(&doc).await?;

        // Insert into Tantivy for full-text search
        self.insert_to_tantivy(&doc).await?;

        Ok(())
    }

    async fn insert_to_lancedb(&self, doc: &VideoDocument) -> Result<()> {
        // Create RecordBatch from document
        let batch = self.create_record_batch(vec![doc.clone()])?;

        // Get or create table
        let table_name = "videos";
        let table = if self.lance_conn.table_exists(table_name).await
            .map_err(|e| VideoRssError::Unknown(format!("Table exists check error: {}", e)))?
        {
            self.lance_conn.open_table(table_name).await
                .map_err(|e| VideoRssError::Unknown(format!("Open table error: {}", e)))?
        } else {
            self.lance_conn.create_table(table_name, batch.clone(), None).await
                .map_err(|e| VideoRssError::Unknown(format!("Create table error: {}", e)))?
        };

        // Add data to table
        table.add(vec![batch]).await
            .map_err(|e| VideoRssError::Unknown(format!("Add to table error: {}", e)))?;

        Ok(())
    }

    async fn insert_to_tantivy(&self, doc: &VideoDocument) -> Result<()> {
        let mut writer = self.index_writer.write().await;

        let mut tantivy_doc = tantivy::Document::new();

        let id_field = self.tantivy_schema.get_field("id").unwrap();
        let title_field = self.tantivy_schema.get_field("title").unwrap();
        let description_field = self.tantivy_schema.get_field("description").unwrap();
        let transcript_field = self.tantivy_schema.get_field("transcript").unwrap();
        let platform_field = self.tantivy_schema.get_field("platform").unwrap();
        let author_field = self.tantivy_schema.get_field("author").unwrap();
        let url_field = self.tantivy_schema.get_field("url").unwrap();
        let upload_timestamp_field = self.tantivy_schema.get_field("upload_timestamp").unwrap();
        let view_count_field = self.tantivy_schema.get_field("view_count").unwrap();
        let duration_field = self.tantivy_schema.get_field("duration_seconds").unwrap();
        let tags_field = self.tantivy_schema.get_field("tags").unwrap();

        tantivy_doc.add_text(id_field, &doc.id);
        tantivy_doc.add_text(title_field, &doc.title);
        tantivy_doc.add_text(description_field, &doc.description);
        tantivy_doc.add_text(transcript_field, &doc.transcript);
        tantivy_doc.add_text(platform_field, &doc.platform);
        tantivy_doc.add_text(author_field, &doc.author);
        tantivy_doc.add_text(url_field, &doc.url);
        tantivy_doc.add_i64(upload_timestamp_field, doc.upload_timestamp);
        tantivy_doc.add_i64(view_count_field, doc.view_count);
        tantivy_doc.add_i64(duration_field, doc.duration_seconds);
        tantivy_doc.add_text(tags_field, &doc.tags.join(" "));

        writer.add_document(tantivy_doc)
            .map_err(|e| VideoRssError::Unknown(format!("Tantivy add document error: {}", e)))?;

        writer.commit()
            .map_err(|e| VideoRssError::Unknown(format!("Tantivy commit error: {}", e)))?;

        Ok(())
    }

    pub async fn hybrid_search(
        &self,
        query: &str,
        embedding: Option<Vec<f32>>,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        // Perform vector search if embedding provided
        if let Some(emb) = embedding {
            let vector_results = self.vector_search(&emb, filters.clone()).await?;
            results.extend(vector_results);
        }

        // Perform text search
        let text_results = self.text_search(query, filters).await?;

        // Merge and deduplicate results
        for text_result in text_results {
            if !results.iter().any(|r| r.document.id == text_result.document.id) {
                results.push(text_result);
            }
        }

        // Re-rank results using reciprocal rank fusion
        self.rerank_results(&mut results);

        // Limit to max_results
        results.truncate(self.config.max_results);

        Ok(results)
    }

    async fn vector_search(
        &self,
        embedding: &[f32],
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        let table = self.lance_conn.open_table("videos").await
            .map_err(|e| VideoRssError::Unknown(format!("Open table error: {}", e)))?;

        // Build query
        let mut query = table
            .query()
            .nearest_to(embedding)
            .map_err(|e| VideoRssError::Unknown(format!("Query build error: {}", e)))?
            .limit(self.config.max_results);

        // Apply filters if provided
        if let Some(f) = filters {
            if let Some(platform) = f.platform {
                query = query.filter(format!("platform = '{}'", platform))
                    .map_err(|e| VideoRssError::Unknown(format!("Filter error: {}", e)))?;
            }
            if let Some(min_views) = f.min_views {
                query = query.filter(format!("view_count >= {}", min_views))
                    .map_err(|e| VideoRssError::Unknown(format!("Filter error: {}", e)))?;
            }
            if let Some(after) = f.uploaded_after {
                query = query.filter(format!("upload_timestamp >= {}", after))
                    .map_err(|e| VideoRssError::Unknown(format!("Filter error: {}", e)))?;
            }
        }

        // Execute query
        let results = query.execute().await
            .map_err(|e| VideoRssError::Unknown(format!("Query execution error: {}", e)))?;

        // Convert results
        let mut search_results = Vec::new();
        for (i, batch) in results.iter().enumerate() {
            if let Ok(doc) = self.record_batch_to_document(batch, i) {
                search_results.push(SearchResult {
                    document: doc,
                    score: 1.0 / (i as f32 + 1.0),  // Simple scoring
                    distance: 0.0,  // Would need to calculate actual distance
                    highlights: Vec::new(),
                });
            }
        }

        Ok(search_results)
    }

    async fn text_search(
        &self,
        query: &str,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        let searcher = self.index_reader.searcher();

        // Build query
        let query_parser = QueryParser::for_index(
            &self.tantivy_index,
            vec![
                self.tantivy_schema.get_field("title").unwrap(),
                self.tantivy_schema.get_field("description").unwrap(),
                self.tantivy_schema.get_field("transcript").unwrap(),
            ],
        );

        let tantivy_query = query_parser.parse_query(query)
            .map_err(|e| VideoRssError::Unknown(format!("Query parse error: {}", e)))?;

        // Execute search
        let top_docs = searcher.search(&tantivy_query, &TopDocs::with_limit(self.config.max_results))
            .map_err(|e| VideoRssError::Unknown(format!("Search error: {}", e)))?;

        // Convert results
        let mut search_results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc = searcher.doc(doc_address)
                .map_err(|e| VideoRssError::Unknown(format!("Doc retrieval error: {}", e)))?;

            if let Ok(video_doc) = self.tantivy_doc_to_video(&retrieved_doc) {
                // Apply filters
                if let Some(ref f) = filters {
                    if !self.apply_filters(&video_doc, f) {
                        continue;
                    }
                }

                search_results.push(SearchResult {
                    document: video_doc,
                    score,
                    distance: 0.0,
                    highlights: self.extract_highlights(query, &retrieved_doc),
                });
            }
        }

        Ok(search_results)
    }

    fn apply_filters(&self, doc: &VideoDocument, filters: &SearchFilters) -> bool {
        if let Some(ref platform) = filters.platform {
            if &doc.platform != platform {
                return false;
            }
        }
        if let Some(min_views) = filters.min_views {
            if doc.view_count < min_views {
                return false;
            }
        }
        if let Some(after) = filters.uploaded_after {
            if doc.upload_timestamp < after {
                return false;
            }
        }
        true
    }

    fn rerank_results(&self, results: &mut Vec<SearchResult>) {
        // Implement reciprocal rank fusion or other reranking strategy
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    }

    fn extract_highlights(&self, query: &str, doc: &tantivy::Document) -> Vec<String> {
        // Extract relevant snippets from the document
        // This is a simplified implementation
        Vec::new()
    }

    fn create_record_batch(&self, docs: Vec<VideoDocument>) -> Result<RecordBatch> {
        // Convert documents to Arrow arrays
        let ids: StringArray = docs.iter().map(|d| Some(d.id.as_str())).collect();
        let titles: StringArray = docs.iter().map(|d| Some(d.title.as_str())).collect();
        let descriptions: StringArray = docs.iter().map(|d| Some(d.description.as_str())).collect();

        // Create embedding array
        let embeddings_flat: Vec<f32> = docs.iter()
            .flat_map(|d| d.embedding.clone())
            .collect();
        let embeddings = Float32Array::from(embeddings_flat);

        // Create other arrays...

        RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(titles),
                Arc::new(descriptions),
                // Add other arrays...
            ],
        ).map_err(|e| VideoRssError::Unknown(format!("RecordBatch creation error: {}", e)))
    }

    fn record_batch_to_document(&self, batch: &RecordBatch, index: usize) -> Result<VideoDocument> {
        // Convert RecordBatch row to VideoDocument
        // This is a simplified implementation
        Ok(VideoDocument {
            id: String::new(),
            title: String::new(),
            description: String::new(),
            transcript: String::new(),
            embedding: Vec::new(),
            platform: String::new(),
            author: String::new(),
            url: String::new(),
            upload_timestamp: 0,
            view_count: 0,
            duration_seconds: 0,
            tags: Vec::new(),
        })
    }

    fn tantivy_doc_to_video(&self, doc: &tantivy::Document) -> Result<VideoDocument> {
        // Convert Tantivy document to VideoDocument
        // This is a simplified implementation
        Ok(VideoDocument {
            id: String::new(),
            title: String::new(),
            description: String::new(),
            transcript: String::new(),
            embedding: Vec::new(),
            platform: String::new(),
            author: String::new(),
            url: String::new(),
            upload_timestamp: 0,
            view_count: 0,
            duration_seconds: 0,
            tags: Vec::new(),
        })
    }

    pub async fn get_statistics(&self) -> Result<DatabaseStats> {
        let table = self.lance_conn.open_table("videos").await
            .map_err(|e| VideoRssError::Unknown(format!("Open table error: {}", e)))?;

        let count = table.count_rows().await
            .map_err(|e| VideoRssError::Unknown(format!("Count error: {}", e)))?;

        Ok(DatabaseStats {
            total_documents: count,
            index_size_bytes: 0,  // Would need to calculate
            last_updated: chrono::Utc::now(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFilters {
    pub platform: Option<String>,
    pub author: Option<String>,
    pub min_views: Option<i64>,
    pub max_duration_seconds: Option<i64>,
    pub uploaded_after: Option<i64>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    pub total_documents: usize,
    pub index_size_bytes: u64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}