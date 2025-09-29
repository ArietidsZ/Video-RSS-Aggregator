"""
Topic Segmentation and Chapter Detection
Segments video content into topical chapters
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import time

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    pipeline,
    BertModel,
    BertTokenizer
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from scipy.signal import find_peaks
import spacy

logger = logging.getLogger(__name__)


@dataclass
class TopicSegment:
    """Represents a topic segment/chapter"""

    topic_id: int
    title: str
    description: str
    start_time: float
    end_time: float
    keywords: List[str]
    confidence: float
    subtopics: List[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_chapter_marker(self) -> str:
        """Convert to YouTube chapter marker format"""
        minutes = int(self.start_time // 60)
        seconds = int(self.start_time % 60)
        return f"{minutes:02d}:{seconds:02d} - {self.title}"


@dataclass
class TopicSegmentationResult:
    """Complete topic segmentation result"""

    segments: List[TopicSegment]
    num_topics: int
    topic_distribution: np.ndarray  # Topic distribution over time
    coherence_score: float
    transition_points: List[float]  # Time points where topics change
    processing_time: float

    def to_chapters(self) -> str:
        """Convert to chapter list format"""
        chapters = []
        for segment in self.segments:
            chapters.append(segment.to_chapter_marker())
        return "\n".join(chapters)

    def get_topic_timeline(self) -> Dict[str, List[Tuple[float, float]]]:
        """Get timeline for each topic"""
        timeline = {}
        for segment in self.segments:
            if segment.title not in timeline:
                timeline[segment.title] = []
            timeline[segment.title].append((segment.start_time, segment.end_time))
        return timeline


class TopicSegmenter:
    """
    Segment video content into topical chapters
    Uses NLP and semantic analysis for topic detection
    """

    def __init__(self,
                 method: str = "semantic",
                 min_segment_duration: float = 30.0,
                 max_segments: int = 20,
                 use_gpu: bool = True,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize topic segmenter

        Args:
            method: Segmentation method ("semantic", "lda", "graph", "hybrid")
            min_segment_duration: Minimum segment duration in seconds
            max_segments: Maximum number of segments
            use_gpu: Use GPU acceleration
            model_name: Transformer model for embeddings
        """
        self.method = method
        self.min_segment_duration = min_segment_duration
        self.max_segments = max_segments
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model_name = model_name

        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # Initialize models
        self._init_models()

        logger.info(f"TopicSegmenter initialized with {method} method")

    def _init_models(self):
        """Initialize NLP models"""
        # Sentence transformer for embeddings
        self.sentence_model = SentenceTransformer(self.model_name, device=self.device)

        # Spacy for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("Spacy model not found, downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        # Summarization pipeline
        if self.use_gpu:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0
            )
        else:
            self.summarizer = None

    def segment(self, transcript: List[Dict[str, Any]],
               video_duration: float,
               use_visual_cues: bool = False,
               scene_boundaries: Optional[List[float]] = None) -> TopicSegmentationResult:
        """
        Segment transcript into topical chapters

        Args:
            transcript: List of transcript segments with text and timestamps
            video_duration: Total video duration
            use_visual_cues: Use scene boundaries as hints
            scene_boundaries: Optional scene boundary timestamps

        Returns:
            Topic segmentation result
        """
        start_time = time.time()

        if not transcript:
            return TopicSegmentationResult(
                segments=[],
                num_topics=0,
                topic_distribution=np.array([]),
                coherence_score=0.0,
                transition_points=[],
                processing_time=0.0
            )

        # Prepare text windows
        text_windows = self._create_text_windows(transcript, video_duration)

        # Perform segmentation based on method
        if self.method == "semantic":
            segments = self._segment_semantic(text_windows)
        elif self.method == "lda":
            segments = self._segment_lda(text_windows)
        elif self.method == "graph":
            segments = self._segment_graph(text_windows)
        elif self.method == "hybrid":
            segments = self._segment_hybrid(text_windows)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Incorporate visual cues if available
        if use_visual_cues and scene_boundaries:
            segments = self._align_with_scenes(segments, scene_boundaries)

        # Post-process segments
        segments = self._post_process_segments(segments, text_windows)

        # Calculate topic distribution
        topic_distribution = self._calculate_topic_distribution(segments, video_duration)

        # Calculate coherence score
        coherence_score = self._calculate_coherence(segments, text_windows)

        # Extract transition points
        transition_points = [seg.start_time for seg in segments[1:]]

        processing_time = time.time() - start_time

        logger.info(f"Segmented into {len(segments)} topics in {processing_time:.2f}s")

        return TopicSegmentationResult(
            segments=segments,
            num_topics=len(segments),
            topic_distribution=topic_distribution,
            coherence_score=coherence_score,
            transition_points=transition_points,
            processing_time=processing_time
        )

    def _create_text_windows(self, transcript: List[Dict[str, Any]],
                            video_duration: float,
                            window_size: float = 30.0) -> List[Dict[str, Any]]:
        """Create fixed-size text windows from transcript"""
        windows = []
        current_window = {"text": [], "start_time": 0, "end_time": 0}

        for segment in transcript:
            text = segment.get("text", "")
            start = segment.get("start_time", 0)
            end = segment.get("end_time", start + 1)

            if not current_window["text"]:
                current_window["start_time"] = start

            current_window["text"].append(text)
            current_window["end_time"] = end

            # Check if window is large enough
            if current_window["end_time"] - current_window["start_time"] >= window_size:
                current_window["text"] = " ".join(current_window["text"])
                windows.append(current_window)
                current_window = {"text": [], "start_time": end, "end_time": end}

        # Add last window
        if current_window["text"]:
            current_window["text"] = " ".join(current_window["text"])
            windows.append(current_window)

        return windows

    def _segment_semantic(self, text_windows: List[Dict[str, Any]]) -> List[TopicSegment]:
        """Segment using semantic similarity"""
        if not text_windows:
            return []

        # Get embeddings for each window
        texts = [w["text"] for w in text_windows]
        embeddings = self.sentence_model.encode(texts, convert_to_tensor=True)

        # Calculate similarity between consecutive windows
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = torch.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[i + 1].unsqueeze(0)
            ).item()
            similarities.append(sim)

        # Find topic boundaries (low similarity points)
        boundaries = self._find_boundaries_from_similarity(similarities)

        # Create segments
        segments = []
        for i, (start_idx, end_idx) in enumerate(boundaries):
            start_time = text_windows[start_idx]["start_time"]
            end_time = text_windows[min(end_idx, len(text_windows) - 1)]["end_time"]

            # Get segment text
            segment_texts = [text_windows[j]["text"] for j in range(start_idx, min(end_idx + 1, len(text_windows)))]
            segment_text = " ".join(segment_texts)

            # Generate title and description
            title, description = self._generate_title_description(segment_text)

            # Extract keywords
            keywords = self._extract_keywords(segment_text)

            segment = TopicSegment(
                topic_id=i,
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                keywords=keywords,
                confidence=0.8
            )
            segments.append(segment)

        return segments

    def _segment_lda(self, text_windows: List[Dict[str, Any]]) -> List[TopicSegment]:
        """Segment using Latent Dirichlet Allocation"""
        if not text_windows:
            return []

        texts = [w["text"] for w in text_windows]

        # Vectorize texts
        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        doc_term_matrix = vectorizer.fit_transform(texts)

        # Determine number of topics
        num_topics = min(self.max_segments, max(2, len(texts) // 5))

        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=100
        )
        doc_topic_dist = lda.fit_transform(doc_term_matrix)

        # Assign windows to topics
        topic_assignments = np.argmax(doc_topic_dist, axis=1)

        # Get topic words
        feature_names = vectorizer.get_feature_names_out()
        topic_words = self._get_lda_topic_words(lda, feature_names)

        # Create segments from consecutive windows with same topic
        segments = []
        current_topic = topic_assignments[0]
        start_idx = 0

        for i in range(1, len(topic_assignments)):
            if topic_assignments[i] != current_topic:
                # Create segment
                segment = self._create_segment_from_windows(
                    text_windows[start_idx:i],
                    current_topic,
                    topic_words[current_topic]
                )
                segments.append(segment)

                # Start new segment
                current_topic = topic_assignments[i]
                start_idx = i

        # Add last segment
        segment = self._create_segment_from_windows(
            text_windows[start_idx:],
            current_topic,
            topic_words[current_topic]
        )
        segments.append(segment)

        return segments

    def _segment_graph(self, text_windows: List[Dict[str, Any]]) -> List[TopicSegment]:
        """Segment using graph-based text segmentation"""
        if not text_windows:
            return []

        texts = [w["text"] for w in text_windows]

        # Create similarity graph
        embeddings = self.sentence_model.encode(texts)
        similarity_matrix = cosine_similarity(embeddings)

        # Create graph
        G = nx.Graph()
        for i in range(len(texts)):
            G.add_node(i)

        # Add edges for similar windows
        threshold = np.percentile(similarity_matrix, 70)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if similarity_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i, j])

        # Find communities (topics)
        communities = nx.community.greedy_modularity_communities(G)

        # Create segments from communities
        segments = []
        for i, community in enumerate(communities):
            indices = sorted(list(community))
            if not indices:
                continue

            start_time = text_windows[indices[0]]["start_time"]
            end_time = text_windows[indices[-1]]["end_time"]

            # Get community text
            community_texts = [text_windows[idx]["text"] for idx in indices]
            community_text = " ".join(community_texts)

            # Generate metadata
            title, description = self._generate_title_description(community_text)
            keywords = self._extract_keywords(community_text)

            segment = TopicSegment(
                topic_id=i,
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                keywords=keywords,
                confidence=0.75
            )
            segments.append(segment)

        # Sort by start time
        segments.sort(key=lambda s: s.start_time)

        return segments

    def _segment_hybrid(self, text_windows: List[Dict[str, Any]]) -> List[TopicSegment]:
        """Hybrid segmentation combining multiple methods"""
        # Get segmentations from different methods
        semantic_segments = self._segment_semantic(text_windows)
        lda_segments = self._segment_lda(text_windows)

        # Merge segmentations
        all_boundaries = set()
        for seg in semantic_segments:
            all_boundaries.add(seg.start_time)
        for seg in lda_segments:
            all_boundaries.add(seg.start_time)

        # Create consensus segments
        boundaries = sorted(list(all_boundaries))
        segments = []

        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]

            # Get text for this segment
            segment_windows = [
                w for w in text_windows
                if w["start_time"] >= start_time and w["end_time"] <= end_time
            ]

            if not segment_windows:
                continue

            segment_text = " ".join([w["text"] for w in segment_windows])

            # Generate metadata
            title, description = self._generate_title_description(segment_text)
            keywords = self._extract_keywords(segment_text)

            segment = TopicSegment(
                topic_id=i,
                title=title,
                description=description,
                start_time=start_time,
                end_time=end_time,
                keywords=keywords,
                confidence=0.85
            )
            segments.append(segment)

        return segments

    def _find_boundaries_from_similarity(self, similarities: List[float]) -> List[Tuple[int, int]]:
        """Find topic boundaries from similarity scores"""
        # Convert to numpy array
        sims = np.array(similarities)

        # Find valleys (low similarity points)
        inverted = 1 - sims
        peaks, properties = find_peaks(inverted, distance=len(sims) // self.max_segments)

        # Add start and end
        boundaries = [0] + list(peaks) + [len(similarities)]

        # Create segments
        segments = []
        for i in range(len(boundaries) - 1):
            segments.append((boundaries[i], boundaries[i + 1]))

        return segments

    def _generate_title_description(self, text: str) -> Tuple[str, str]:
        """Generate title and description for segment"""
        # Use NLP to extract key phrases
        doc = self.nlp(text[:1000])  # Limit text length

        # Extract noun phrases as potential titles
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        if noun_phrases:
            title = max(noun_phrases, key=len)[:50]  # Use longest noun phrase
        else:
            title = "Topic Segment"

        # Generate description
        if self.summarizer and len(text) > 100:
            try:
                summary = self.summarizer(text, max_length=50, min_length=10, do_sample=False)
                description = summary[0]["summary_text"]
            except:
                description = text[:200] + "..."
        else:
            description = text[:200] + "..."

        return title, description

    def _extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """Extract keywords from text"""
        doc = self.nlp(text[:5000])  # Limit text length

        # Extract entities
        entities = [ent.text.lower() for ent in doc.ents]

        # Extract noun phrases
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

        # Count frequency
        from collections import Counter
        word_freq = Counter(entities + noun_phrases)

        # Get top keywords
        keywords = [word for word, _ in word_freq.most_common(num_keywords)]

        return keywords

    def _get_lda_topic_words(self, lda_model, feature_names, num_words: int = 5) -> Dict[int, List[str]]:
        """Get top words for each LDA topic"""
        topic_words = {}

        for topic_idx, topic in enumerate(lda_model.components_):
            top_word_indices = topic.argsort()[-num_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topic_words[topic_idx] = top_words

        return topic_words

    def _create_segment_from_windows(self, windows: List[Dict[str, Any]],
                                    topic_id: int,
                                    topic_words: List[str]) -> TopicSegment:
        """Create segment from windows"""
        if not windows:
            return None

        start_time = windows[0]["start_time"]
        end_time = windows[-1]["end_time"]

        # Combine text
        text = " ".join([w["text"] for w in windows])

        # Generate title from topic words
        title = " ".join(topic_words[:3]).title() if topic_words else f"Topic {topic_id}"

        # Generate description
        description = text[:200] + "..." if len(text) > 200 else text

        return TopicSegment(
            topic_id=topic_id,
            title=title,
            description=description,
            start_time=start_time,
            end_time=end_time,
            keywords=topic_words,
            confidence=0.7
        )

    def _align_with_scenes(self, segments: List[TopicSegment],
                          scene_boundaries: List[float]) -> List[TopicSegment]:
        """Align topic segments with scene boundaries"""
        aligned_segments = []

        for segment in segments:
            # Find nearest scene boundary to segment start
            nearest_start = min(scene_boundaries, key=lambda x: abs(x - segment.start_time))

            # Find nearest scene boundary to segment end
            nearest_end = min(scene_boundaries, key=lambda x: abs(x - segment.end_time))

            # Only adjust if boundaries are close (within 5 seconds)
            if abs(nearest_start - segment.start_time) < 5.0:
                segment.start_time = nearest_start

            if abs(nearest_end - segment.end_time) < 5.0:
                segment.end_time = nearest_end

            aligned_segments.append(segment)

        return aligned_segments

    def _post_process_segments(self, segments: List[TopicSegment],
                              text_windows: List[Dict[str, Any]]) -> List[TopicSegment]:
        """Post-process segments"""
        if not segments:
            return segments

        # Merge short segments
        processed = []
        current_segment = segments[0]

        for next_segment in segments[1:]:
            if current_segment.duration < self.min_segment_duration:
                # Merge with next segment
                current_segment.end_time = next_segment.end_time
                current_segment.keywords.extend(next_segment.keywords)
                current_segment.keywords = list(set(current_segment.keywords))[:10]
            else:
                processed.append(current_segment)
                current_segment = next_segment

        processed.append(current_segment)

        # Ensure no gaps
        for i in range(len(processed) - 1):
            processed[i].end_time = processed[i + 1].start_time

        return processed

    def _calculate_topic_distribution(self, segments: List[TopicSegment],
                                     video_duration: float) -> np.ndarray:
        """Calculate topic distribution over time"""
        # Create time grid
        time_resolution = 1.0  # 1 second resolution
        num_points = int(video_duration / time_resolution)
        distribution = np.zeros((len(segments), num_points))

        for topic_idx, segment in enumerate(segments):
            start_idx = int(segment.start_time / time_resolution)
            end_idx = min(int(segment.end_time / time_resolution), num_points)
            distribution[topic_idx, start_idx:end_idx] = 1.0

        return distribution

    def _calculate_coherence(self, segments: List[TopicSegment],
                           text_windows: List[Dict[str, Any]]) -> float:
        """Calculate topic coherence score"""
        if not segments or not text_windows:
            return 0.0

        # Simple coherence based on keyword overlap
        coherence_scores = []

        for i in range(len(segments) - 1):
            keywords1 = set(segments[i].keywords)
            keywords2 = set(segments[i + 1].keywords)

            if keywords1 and keywords2:
                overlap = len(keywords1.intersection(keywords2))
                total = len(keywords1.union(keywords2))
                coherence = 1 - (overlap / total)  # Less overlap = more distinct topics
                coherence_scores.append(coherence)

        return np.mean(coherence_scores) if coherence_scores else 0.5