use crate::{
    bilibili::BilibiliClient,
    content::ContentAnalyzer,
    rss::RssGenerator,
    types::*,
    Result,
};
use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct PyBilibiliClient {
    client: BilibiliClient,
}

#[pymethods]
impl PyBilibiliClient {
    #[new]
    #[pyo3(signature = (credentials=None))]
    pub fn new(credentials: Option<HashMap<String, String>>) -> PyResult<Self> {
        let rust_credentials = credentials.map(|creds| BilibiliCredentials {
            sessdata: creds.get("sessdata").cloned().unwrap_or_default(),
            bili_jct: creds.get("bili_jct").cloned().unwrap_or_default(),
            buvid3: creds.get("buvid3").cloned().unwrap_or_default(),
        });

        let client = BilibiliClient::new(rust_credentials)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { client })
    }

    #[pyo3(signature = (limit=10, include_transcription=false, personalized=false))]
    pub fn fetch_recommendations<'py>(
        &self,
        py: Python<'py>,
        limit: usize,
        include_transcription: bool,
        personalized: bool,
    ) -> PyResult<&'py PyAny> {
        let client = &self.client;
        let options = FetchOptions {
            limit,
            include_transcription,
            personalized,
            credentials: None, // Already set in client
        };

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let videos = client
                .fetch_recommendations(&options)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let py_videos: Vec<HashMap<String, PyObject>> = Python::with_gil(|py| {
                videos
                    .into_iter()
                    .map(|video| video_info_to_py_dict(py, video))
                    .collect()
            });

            Ok(py_videos)
        })
    }

    pub fn fetch_video_info<'py>(
        &self,
        py: Python<'py>,
        video_url: &str,
    ) -> PyResult<&'py PyAny> {
        let client = &self.client;
        let url = video_url.to_string();

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let video = client
                .fetch_video_info(&url)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let py_video = Python::with_gil(|py| video_info_to_py_dict(py, video));

            Ok(py_video)
        })
    }

    pub fn get_video_subtitles<'py>(
        &self,
        py: Python<'py>,
        video_id: &str,
    ) -> PyResult<&'py PyAny> {
        let client = &self.client;
        let id = video_id.to_string();

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let subtitles = client
                .get_video_subtitles(&id)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            Ok(subtitles)
        })
    }
}

#[pyclass]
pub struct PyRssGenerator {
    generator: RssGenerator,
}

#[pymethods]
impl PyRssGenerator {
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<HashMap<String, String>>) -> Self {
        let rust_config = config
            .map(|cfg| RssConfig {
                title: cfg.get("title").cloned().unwrap_or_else(|| "AI智能内容摘要".to_string()),
                description: cfg.get("description").cloned().unwrap_or_else(|| "AI视频聚合".to_string()),
                link: cfg.get("link").cloned().unwrap_or_else(|| "http://localhost:3000".to_string()),
                language: cfg.get("language").cloned().unwrap_or_else(|| "zh-CN".to_string()),
                generator: cfg.get("generator").cloned().unwrap_or_else(|| "Rust RSS Core".to_string()),
            })
            .unwrap_or_default();

        let generator = RssGenerator::new(rust_config);

        Self { generator }
    }

    pub fn generate_feed(&self, videos: Vec<HashMap<String, PyObject>>) -> PyResult<String> {
        Python::with_gil(|py| {
            let rust_videos: Result<Vec<VideoInfo>, PyErr> = videos
                .into_iter()
                .map(|py_video| py_dict_to_video_info(py, py_video))
                .collect();

            let rust_videos = rust_videos?;

            self.generator
                .generate_feed(&rust_videos)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }

    #[pyo3(signature = (videos, include_ai_summary=true))]
    pub fn generate_feed_with_summary(
        &self,
        videos: Vec<HashMap<String, PyObject>>,
        include_ai_summary: bool,
    ) -> PyResult<String> {
        Python::with_gil(|py| {
            let rust_videos: Result<Vec<VideoInfo>, PyErr> = videos
                .into_iter()
                .map(|py_video| py_dict_to_video_info(py, py_video))
                .collect();

            let rust_videos = rust_videos?;

            self.generator
                .generate_feed_with_summary(&rust_videos, include_ai_summary)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        })
    }
}

#[pyclass]
pub struct PyContentAnalyzer {
    analyzer: ContentAnalyzer,
}

#[pymethods]
impl PyContentAnalyzer {
    #[new]
    pub fn new() -> PyResult<Self> {
        let analyzer = ContentAnalyzer::new()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { analyzer })
    }

    pub fn analyze_video<'py>(
        &self,
        py: Python<'py>,
        video: HashMap<String, PyObject>,
    ) -> PyResult<&'py PyAny> {
        let analyzer = &self.analyzer;
        let rust_video = py_dict_to_video_info(py, video)?;

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let summary = analyzer
                .analyze_video(&rust_video)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let py_summary = Python::with_gil(|py| content_summary_to_py_dict(py, summary));

            Ok(py_summary)
        })
    }

    pub fn batch_analyze<'py>(
        &self,
        py: Python<'py>,
        videos: Vec<HashMap<String, PyObject>>,
    ) -> PyResult<&'py PyAny> {
        let analyzer = &self.analyzer;

        let rust_videos: Result<Vec<VideoInfo>, PyErr> = videos
            .into_iter()
            .map(|py_video| py_dict_to_video_info(py, py_video))
            .collect();

        let rust_videos = rust_videos?;

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let results = analyzer
                .batch_analyze(&rust_videos)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let py_results: Vec<HashMap<String, PyObject>> = Python::with_gil(|py| {
                results
                    .into_iter()
                    .map(|result| analysis_result_to_py_dict(py, result))
                    .collect()
            });

            Ok(py_results)
        })
    }
}

// Helper functions for conversion between Python and Rust types

fn video_info_to_py_dict(py: Python, video: VideoInfo) -> HashMap<String, PyObject> {
    let mut dict = HashMap::new();

    dict.insert("id".to_string(), video.id.to_object(py));
    dict.insert("title".to_string(), video.title.to_object(py));
    dict.insert("description".to_string(), video.description.to_object(py));
    dict.insert("url".to_string(), video.url.to_object(py));
    dict.insert("author".to_string(), video.author.to_object(py));
    dict.insert("upload_date".to_string(), video.upload_date.to_rfc3339().to_object(py));
    dict.insert("duration".to_string(), video.duration.to_object(py));
    dict.insert("view_count".to_string(), video.view_count.to_object(py));
    dict.insert("like_count".to_string(), video.like_count.to_object(py));
    dict.insert("comment_count".to_string(), video.comment_count.to_object(py));
    dict.insert("tags".to_string(), video.tags.to_object(py));
    dict.insert("thumbnail_url".to_string(), video.thumbnail_url.to_object(py));
    dict.insert("platform".to_string(), video.platform.as_str().to_object(py));

    if let Some(transcription) = video.transcription {
        dict.insert("transcription".to_string(), transcription_data_to_py_dict(py, transcription).to_object(py));
    }

    dict
}

fn py_dict_to_video_info(py: Python, dict: HashMap<String, PyObject>) -> PyResult<VideoInfo> {
    use chrono::DateTime;

    let id = dict.get("id")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'id' field"))?
        .extract::<String>(py)?;

    let title = dict.get("title")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'title' field"))?
        .extract::<String>(py)?;

    let description = dict.get("description")
        .map(|obj| obj.extract::<String>(py))
        .transpose()?
        .unwrap_or_default();

    let url = dict.get("url")
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' field"))?
        .extract::<String>(py)?;

    let author = dict.get("author")
        .map(|obj| obj.extract::<String>(py))
        .transpose()?
        .unwrap_or_default();

    let upload_date = dict.get("upload_date")
        .map(|obj| obj.extract::<String>(py))
        .transpose()?
        .and_then(|date_str| DateTime::parse_from_rfc3339(&date_str).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(chrono::Utc::now);

    let platform_str = dict.get("platform")
        .map(|obj| obj.extract::<String>(py))
        .transpose()?
        .unwrap_or_else(|| "bilibili".to_string());

    let platform = match platform_str.as_str() {
        "bilibili" => Platform::Bilibili,
        "douyin" => Platform::Douyin,
        "kuaishou" => Platform::Kuaishou,
        "youtube" => Platform::YouTube,
        _ => Platform::Bilibili,
    };

    Ok(VideoInfo {
        id,
        title,
        description,
        url,
        author,
        upload_date,
        duration: dict.get("duration").and_then(|obj| obj.extract::<u64>(py).ok()),
        view_count: dict.get("view_count").and_then(|obj| obj.extract::<u64>(py).ok()).unwrap_or(0),
        like_count: dict.get("like_count").and_then(|obj| obj.extract::<u64>(py).ok()).unwrap_or(0),
        comment_count: dict.get("comment_count").and_then(|obj| obj.extract::<u64>(py).ok()).unwrap_or(0),
        tags: dict.get("tags").and_then(|obj| obj.extract::<Vec<String>>(py).ok()).unwrap_or_default(),
        thumbnail_url: dict.get("thumbnail_url").and_then(|obj| obj.extract::<String>(py).ok()),
        platform,
        transcription: None, // TODO: Handle transcription conversion
    })
}

fn transcription_data_to_py_dict(py: Python, transcription: TranscriptionData) -> HashMap<String, PyObject> {
    let mut dict = HashMap::new();

    dict.insert("paragraph_summary".to_string(), transcription.paragraph_summary.to_object(py));
    dict.insert("sentence_subtitle".to_string(), transcription.sentence_subtitle.to_object(py));
    dict.insert("full_transcript".to_string(), transcription.full_transcript.to_object(py));
    dict.insert("status".to_string(), format!("{:?}", transcription.status).to_object(py));

    let mut model_info = HashMap::new();
    model_info.insert("transcriber".to_string(), transcription.model_info.transcriber.to_object(py));
    model_info.insert("summarizer".to_string(), transcription.model_info.summarizer.to_object(py));
    dict.insert("model_info".to_string(), model_info.to_object(py));

    dict.insert("source_types".to_string(), transcription.source_types.to_object(py));

    dict
}

fn content_summary_to_py_dict(py: Python, summary: ContentSummary) -> HashMap<String, PyObject> {
    let mut dict = HashMap::new();

    dict.insert("ai_summary".to_string(), summary.ai_summary.to_object(py));
    dict.insert("keywords".to_string(), summary.keywords.to_object(py));
    dict.insert("sentiment".to_string(), summary.sentiment.to_object(py));
    dict.insert("content_type".to_string(), format!("{:?}", summary.content_type).to_object(py));

    dict
}

fn analysis_result_to_py_dict(py: Python, result: AnalysisResult) -> HashMap<String, PyObject> {
    let mut dict = HashMap::new();

    dict.insert("video".to_string(), video_info_to_py_dict(py, result.video).to_object(py));
    dict.insert("summary".to_string(), content_summary_to_py_dict(py, result.summary).to_object(py));
    dict.insert("processing_time_ms".to_string(), result.processing_time_ms.to_object(py));

    dict
}

#[pymodule]
fn video_rss_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBilibiliClient>()?;
    m.add_class::<PyRssGenerator>()?;
    m.add_class::<PyContentAnalyzer>()?;
    Ok(())
}