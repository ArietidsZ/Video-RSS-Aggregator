"""
Speaker Diarization for Multi-Speaker Videos
Identifies and segments different speakers in audio
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import time
import json

import numpy as np
import torch
from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Segment, Annotation
import librosa
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from resemblyzer import VoiceEncoder, preprocess_wav

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Represents a speaker segment"""

    speaker_id: str
    start_time: float
    end_time: float
    confidence: float
    embedding: Optional[np.ndarray] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def overlaps(self, other: "SpeakerSegment") -> bool:
        return not (self.end_time <= other.start_time or self.start_time >= other.end_time)


@dataclass
class SpeakerProfile:
    """Profile of a speaker"""

    speaker_id: str
    total_duration: float
    segment_count: int
    avg_segment_duration: float
    embeddings: List[np.ndarray]
    dominant_frequency: Optional[float] = None
    speaking_rate: Optional[float] = None

    def get_average_embedding(self) -> np.ndarray:
        """Get average speaker embedding"""
        if self.embeddings:
            return np.mean(self.embeddings, axis=0)
        return np.array([])


@dataclass
class DiarizationResult:
    """Complete diarization result"""

    segments: List[SpeakerSegment]
    speakers: List[SpeakerProfile]
    num_speakers: int
    total_duration: float
    speaking_time_ratio: float
    overlap_ratio: float
    processing_time: float

    def get_speaker_timeline(self) -> Dict[str, List[Tuple[float, float]]]:
        """Get timeline for each speaker"""
        timeline = {}
        for segment in self.segments:
            if segment.speaker_id not in timeline:
                timeline[segment.speaker_id] = []
            timeline[segment.speaker_id].append((segment.start_time, segment.end_time))
        return timeline

    def to_rttm(self) -> str:
        """Convert to RTTM format"""
        lines = []
        for segment in self.segments:
            lines.append(
                f"SPEAKER file 1 {segment.start_time:.3f} {segment.duration:.3f} "
                f"<NA> <NA> {segment.speaker_id} <NA> <NA>"
            )
        return "\n".join(lines)


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio and resemblyzer
    Identifies who spoke when in multi-speaker videos
    """

    def __init__(self,
                 model_type: str = "pyannote",
                 num_speakers: Optional[int] = None,
                 min_speakers: int = 1,
                 max_speakers: int = 10,
                 use_gpu: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialize speaker diarizer

        Args:
            model_type: Model to use ("pyannote", "resemblyzer", "ensemble")
            num_speakers: Known number of speakers (None for auto-detection)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            use_gpu: Use GPU acceleration
            cache_dir: Directory for caching results
        """
        self.model_type = model_type
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # Initialize models
        self._init_models()

        logger.info(f"SpeakerDiarizer initialized with {model_type} model")

    def _init_models(self):
        """Initialize diarization models"""
        if self.model_type in ["pyannote", "ensemble"]:
            try:
                # Load pretrained pipeline
                self.pyannote_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=None  # Add token if needed
                )
                if self.use_gpu:
                    self.pyannote_pipeline.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to load pyannote model: {e}")
                self.pyannote_pipeline = None

        if self.model_type in ["resemblyzer", "ensemble"]:
            try:
                # Load resemblyzer encoder
                self.voice_encoder = VoiceEncoder(device="cuda" if self.use_gpu else "cpu")
            except Exception as e:
                logger.warning(f"Failed to load resemblyzer: {e}")
                self.voice_encoder = None

    def diarize(self, audio_path: str,
               reference_embeddings: Optional[Dict[str, np.ndarray]] = None) -> DiarizationResult:
        """
        Perform speaker diarization on audio

        Args:
            audio_path: Path to audio file
            reference_embeddings: Optional reference embeddings for known speakers

        Returns:
            Diarization result
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        start_time = time.time()

        # Check cache
        if self.cache_dir:
            cache_key = f"{audio_path.stem}_{self.model_type}_{self.num_speakers}"
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                logger.info(f"Loading cached results from {cache_file}")
                return self._load_from_cache(cache_file)

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr

        # Perform diarization based on model type
        if self.model_type == "pyannote":
            result = self._diarize_pyannote(audio_path, duration)
        elif self.model_type == "resemblyzer":
            result = self._diarize_resemblyzer(audio, sr, duration)
        elif self.model_type == "ensemble":
            result = self._diarize_ensemble(audio_path, audio, sr, duration)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Apply reference embeddings if provided
        if reference_embeddings and result:
            result = self._match_reference_speakers(result, reference_embeddings)

        # Calculate statistics
        if result:
            result.processing_time = time.time() - start_time
            result.speaking_time_ratio = self._calculate_speaking_ratio(result)
            result.overlap_ratio = self._calculate_overlap_ratio(result)

            # Cache results
            if self.cache_dir and cache_file:
                self._save_to_cache(result, cache_file)

            logger.info(
                f"Diarization complete: {result.num_speakers} speakers found "
                f"in {result.processing_time:.2f}s"
            )

        return result

    def _diarize_pyannote(self, audio_path: Path, duration: float) -> Optional[DiarizationResult]:
        """Perform diarization using pyannote"""
        if not self.pyannote_pipeline:
            return None

        try:
            # Run pipeline
            params = {
                "min_speakers": self.min_speakers,
                "max_speakers": self.max_speakers
            }
            if self.num_speakers:
                params["num_speakers"] = self.num_speakers

            annotation = self.pyannote_pipeline(str(audio_path), **params)

            # Convert to segments
            segments = []
            for segment, track, label in annotation.itertracks(yield_label=True):
                speaker_segment = SpeakerSegment(
                    speaker_id=label,
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=1.0  # Pyannote doesn't provide confidence
                )
                segments.append(speaker_segment)

            # Create speaker profiles
            speakers = self._create_speaker_profiles(segments, duration)

            return DiarizationResult(
                segments=segments,
                speakers=speakers,
                num_speakers=len(speakers),
                total_duration=duration,
                speaking_time_ratio=0,  # Calculated later
                overlap_ratio=0,  # Calculated later
                processing_time=0  # Set later
            )

        except Exception as e:
            logger.error(f"Pyannote diarization failed: {e}")
            return None

    def _diarize_resemblyzer(self, audio: np.ndarray,
                            sr: int,
                            duration: float) -> Optional[DiarizationResult]:
        """Perform diarization using resemblyzer"""
        if not self.voice_encoder:
            return None

        try:
            # Preprocess audio
            wav = preprocess_wav(audio, sr)

            # Extract embeddings with sliding window
            window_size = int(1.6 * sr)  # 1.6 second windows
            hop_size = int(0.8 * sr)  # 0.8 second hop

            embeddings = []
            timestamps = []

            for start in range(0, len(wav) - window_size, hop_size):
                end = start + window_size
                window = wav[start:end]

                # Get embedding
                embedding = self.voice_encoder.embed_utterance(window)
                embeddings.append(embedding)
                timestamps.append(start / sr)

            embeddings = np.array(embeddings)

            # Cluster embeddings
            if self.num_speakers:
                clustering = AgglomerativeClustering(
                    n_clusters=self.num_speakers,
                    affinity="cosine",
                    linkage="average"
                )
            else:
                # Use DBSCAN for unknown number of speakers
                clustering = DBSCAN(eps=0.3, min_samples=3, metric="cosine")

            labels = clustering.fit_predict(embeddings)

            # Handle DBSCAN noise points
            if isinstance(clustering, DBSCAN):
                # Assign noise points to nearest cluster
                noise_mask = labels == -1
                if np.any(noise_mask):
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
                    nn.fit(embeddings[~noise_mask])
                    _, indices = nn.kneighbors(embeddings[noise_mask])
                    labels[noise_mask] = labels[~noise_mask][indices.flatten()]

            # Create segments
            segments = []
            current_speaker = labels[0]
            segment_start = timestamps[0]

            for i in range(1, len(labels)):
                if labels[i] != current_speaker:
                    # End current segment
                    segments.append(SpeakerSegment(
                        speaker_id=f"speaker_{current_speaker}",
                        start_time=segment_start,
                        end_time=timestamps[i],
                        confidence=0.9,
                        embedding=embeddings[i-1]
                    ))

                    # Start new segment
                    current_speaker = labels[i]
                    segment_start = timestamps[i]

            # Add last segment
            segments.append(SpeakerSegment(
                speaker_id=f"speaker_{current_speaker}",
                start_time=segment_start,
                end_time=duration,
                confidence=0.9,
                embedding=embeddings[-1]
            ))

            # Create speaker profiles
            speakers = self._create_speaker_profiles(segments, duration)

            return DiarizationResult(
                segments=segments,
                speakers=speakers,
                num_speakers=len(speakers),
                total_duration=duration,
                speaking_time_ratio=0,
                overlap_ratio=0,
                processing_time=0
            )

        except Exception as e:
            logger.error(f"Resemblyzer diarization failed: {e}")
            return None

    def _diarize_ensemble(self, audio_path: Path,
                         audio: np.ndarray,
                         sr: int,
                         duration: float) -> Optional[DiarizationResult]:
        """Perform ensemble diarization combining multiple methods"""
        results = []

        # Try pyannote
        pyannote_result = self._diarize_pyannote(audio_path, duration)
        if pyannote_result:
            results.append(pyannote_result)

        # Try resemblyzer
        resemblyzer_result = self._diarize_resemblyzer(audio, sr, duration)
        if resemblyzer_result:
            results.append(resemblyzer_result)

        if not results:
            return None

        if len(results) == 1:
            return results[0]

        # Merge results
        return self._merge_diarization_results(results, duration)

    def _merge_diarization_results(self, results: List[DiarizationResult],
                                  duration: float) -> DiarizationResult:
        """Merge multiple diarization results"""
        # Simple voting-based merging
        time_resolution = 0.1  # 100ms resolution
        num_frames = int(duration / time_resolution)

        # Create voting matrix
        votes = []
        for result in results:
            frame_labels = np.zeros(num_frames, dtype=int)

            for segment in result.segments:
                start_frame = int(segment.start_time / time_resolution)
                end_frame = min(int(segment.end_time / time_resolution), num_frames)
                speaker_id = int(segment.speaker_id.split("_")[-1])
                frame_labels[start_frame:end_frame] = speaker_id

            votes.append(frame_labels)

        votes = np.array(votes)

        # Majority voting
        from scipy.stats import mode
        final_labels, _ = mode(votes, axis=0)
        final_labels = final_labels.flatten()

        # Convert back to segments
        segments = []
        current_speaker = final_labels[0]
        segment_start = 0

        for i in range(1, len(final_labels)):
            if final_labels[i] != current_speaker:
                if current_speaker > 0:  # Ignore silence (label 0)
                    segments.append(SpeakerSegment(
                        speaker_id=f"speaker_{current_speaker}",
                        start_time=segment_start * time_resolution,
                        end_time=i * time_resolution,
                        confidence=0.95
                    ))

                current_speaker = final_labels[i]
                segment_start = i

        # Add last segment
        if current_speaker > 0:
            segments.append(SpeakerSegment(
                speaker_id=f"speaker_{current_speaker}",
                start_time=segment_start * time_resolution,
                end_time=duration,
                confidence=0.95
            ))

        # Create speaker profiles
        speakers = self._create_speaker_profiles(segments, duration)

        return DiarizationResult(
            segments=segments,
            speakers=speakers,
            num_speakers=len(speakers),
            total_duration=duration,
            speaking_time_ratio=0,
            overlap_ratio=0,
            processing_time=0
        )

    def _create_speaker_profiles(self, segments: List[SpeakerSegment],
                                total_duration: float) -> List[SpeakerProfile]:
        """Create speaker profiles from segments"""
        profiles = {}

        for segment in segments:
            if segment.speaker_id not in profiles:
                profiles[segment.speaker_id] = {
                    "duration": 0,
                    "count": 0,
                    "embeddings": []
                }

            profiles[segment.speaker_id]["duration"] += segment.duration
            profiles[segment.speaker_id]["count"] += 1
            if segment.embedding is not None:
                profiles[segment.speaker_id]["embeddings"].append(segment.embedding)

        # Create profile objects
        speaker_profiles = []
        for speaker_id, data in profiles.items():
            profile = SpeakerProfile(
                speaker_id=speaker_id,
                total_duration=data["duration"],
                segment_count=data["count"],
                avg_segment_duration=data["duration"] / data["count"] if data["count"] > 0 else 0,
                embeddings=data["embeddings"]
            )
            speaker_profiles.append(profile)

        return speaker_profiles

    def _calculate_speaking_ratio(self, result: DiarizationResult) -> float:
        """Calculate ratio of speaking time to total duration"""
        speaking_time = sum(seg.duration for seg in result.segments)
        return speaking_time / result.total_duration if result.total_duration > 0 else 0

    def _calculate_overlap_ratio(self, result: DiarizationResult) -> float:
        """Calculate ratio of overlapping segments"""
        overlap_time = 0

        for i, seg1 in enumerate(result.segments):
            for seg2 in result.segments[i+1:]:
                if seg1.overlaps(seg2):
                    overlap_start = max(seg1.start_time, seg2.start_time)
                    overlap_end = min(seg1.end_time, seg2.end_time)
                    overlap_time += overlap_end - overlap_start

        return overlap_time / result.total_duration if result.total_duration > 0 else 0

    def _match_reference_speakers(self, result: DiarizationResult,
                                 reference_embeddings: Dict[str, np.ndarray]) -> DiarizationResult:
        """Match detected speakers to reference speakers"""
        if not result.speakers:
            return result

        # Get average embeddings for each detected speaker
        detected_embeddings = {}
        for speaker in result.speakers:
            avg_embedding = speaker.get_average_embedding()
            if avg_embedding.size > 0:
                detected_embeddings[speaker.speaker_id] = avg_embedding

        if not detected_embeddings:
            return result

        # Match to reference speakers
        speaker_mapping = {}
        for detected_id, detected_emb in detected_embeddings.items():
            best_match = None
            best_similarity = -1

            for ref_id, ref_emb in reference_embeddings.items():
                # Cosine similarity
                similarity = np.dot(detected_emb, ref_emb) / (
                    np.linalg.norm(detected_emb) * np.linalg.norm(ref_emb)
                )

                if similarity > best_similarity and similarity > 0.7:  # Threshold
                    best_similarity = similarity
                    best_match = ref_id

            if best_match:
                speaker_mapping[detected_id] = best_match

        # Update speaker IDs in segments and profiles
        for segment in result.segments:
            if segment.speaker_id in speaker_mapping:
                segment.speaker_id = speaker_mapping[segment.speaker_id]

        for speaker in result.speakers:
            if speaker.speaker_id in speaker_mapping:
                speaker.speaker_id = speaker_mapping[speaker.speaker_id]

        return result

    def _save_to_cache(self, result: DiarizationResult, cache_file: Path):
        """Save results to cache"""
        data = {
            "segments": [
                {
                    "speaker_id": seg.speaker_id,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "confidence": seg.confidence
                }
                for seg in result.segments
            ],
            "num_speakers": result.num_speakers,
            "total_duration": result.total_duration,
            "speaking_time_ratio": result.speaking_time_ratio,
            "overlap_ratio": result.overlap_ratio,
            "processing_time": result.processing_time
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_from_cache(self, cache_file: Path) -> DiarizationResult:
        """Load results from cache"""
        with open(cache_file) as f:
            data = json.load(f)

        segments = [
            SpeakerSegment(**seg) for seg in data["segments"]
        ]

        speakers = self._create_speaker_profiles(segments, data["total_duration"])

        return DiarizationResult(
            segments=segments,
            speakers=speakers,
            num_speakers=data["num_speakers"],
            total_duration=data["total_duration"],
            speaking_time_ratio=data["speaking_time_ratio"],
            overlap_ratio=data["overlap_ratio"],
            processing_time=data["processing_time"]
        )