"""
Subtitle Extraction and Alignment
Extracts embedded subtitles and aligns with audio
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import time
import re
import subprocess
import json

import numpy as np
import webvtt
import srt
import pysubs2
import ffmpeg
from pytesseract import pytesseract
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SubtitleSegment:
    """Single subtitle segment"""

    text: str
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    confidence: float  # OCR confidence if applicable
    language: str      # Language code
    speaker: Optional[str] = None  # Speaker ID if available

    @property
    def duration(self) -> float:
        """Get segment duration"""
        return self.end_time - self.start_time

    def overlaps(self, other: "SubtitleSegment") -> bool:
        """Check if this segment overlaps with another"""
        return not (self.end_time <= other.start_time or self.start_time >= other.end_time)


@dataclass
class SubtitleTrack:
    """Complete subtitle track"""

    segments: List[SubtitleSegment]
    language: str
    format: str  # srt, vtt, ass, etc.
    is_embedded: bool  # True if extracted from video
    is_auto_generated: bool  # True if auto-generated
    total_duration: float
    word_count: int
    avg_reading_speed: float  # Words per minute

    def to_srt(self) -> str:
        """Convert to SRT format"""
        subs = []
        for i, seg in enumerate(self.segments, 1):
            sub = srt.Subtitle(
                index=i,
                start=srt.timedelta(seconds=seg.start_time),
                end=srt.timedelta(seconds=seg.end_time),
                content=seg.text
            )
            subs.append(sub)
        return srt.compose(subs)

    def to_vtt(self) -> str:
        """Convert to WebVTT format"""
        vtt = webvtt.WebVTT()
        for seg in self.segments:
            caption = webvtt.Caption(
                start=self._seconds_to_vtt_time(seg.start_time),
                end=self._seconds_to_vtt_time(seg.end_time),
                text=seg.text
            )
            vtt.captions.append(caption)
        return str(vtt)

    @staticmethod
    def _seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to WebVTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class SubtitleExtractor:
    """
    Extract and process subtitles from videos
    Supports embedded, external, and OCR-based extraction
    """

    def __init__(self,
                 preferred_languages: List[str] = ["en", "zh", "es", "fr"],
                 use_ocr: bool = False,
                 ocr_confidence_threshold: float = 0.7,
                 cache_dir: Optional[str] = None):
        """
        Initialize subtitle extractor

        Args:
            preferred_languages: Preferred subtitle languages
            use_ocr: Enable OCR for hardcoded subtitles
            ocr_confidence_threshold: Minimum OCR confidence
            cache_dir: Directory for caching results
        """
        self.preferred_languages = preferred_languages
        self.use_ocr = use_ocr
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure Tesseract for OCR
        if self.use_ocr:
            self._configure_tesseract()

        logger.info(f"SubtitleExtractor initialized, OCR: {use_ocr}")

    def _configure_tesseract(self):
        """Configure Tesseract OCR"""
        try:
            # Test Tesseract installation
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self.use_ocr = False

    def extract(self, video_path: str,
               extract_all: bool = False,
               target_language: Optional[str] = None) -> List[SubtitleTrack]:
        """
        Extract subtitles from video

        Args:
            video_path: Path to video file
            extract_all: Extract all available subtitle tracks
            target_language: Target language for extraction

        Returns:
            List of subtitle tracks
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        tracks = []

        # Extract embedded subtitles
        embedded_tracks = self._extract_embedded_subtitles(video_path, extract_all, target_language)
        tracks.extend(embedded_tracks)

        # Look for external subtitle files
        external_tracks = self._find_external_subtitles(video_path)
        tracks.extend(external_tracks)

        # OCR extraction if enabled and no subtitles found
        if self.use_ocr and not tracks:
            logger.info("No subtitles found, attempting OCR extraction")
            ocr_track = self._extract_ocr_subtitles(video_path)
            if ocr_track:
                tracks.append(ocr_track)

        # Sort by language preference
        tracks = self._sort_by_language_preference(tracks)

        logger.info(f"Extracted {len(tracks)} subtitle tracks from {video_path.name}")
        return tracks

    def _extract_embedded_subtitles(self, video_path: Path,
                                   extract_all: bool,
                                   target_language: Optional[str]) -> List[SubtitleTrack]:
        """Extract embedded subtitle streams from video"""
        tracks = []

        try:
            # Get stream information
            probe = ffmpeg.probe(str(video_path))
            subtitle_streams = [
                s for s in probe["streams"]
                if s["codec_type"] == "subtitle"
            ]

            for stream in subtitle_streams:
                stream_index = stream["index"]
                language = stream.get("tags", {}).get("language", "und")

                # Skip if not target language
                if target_language and language != target_language:
                    continue

                # Skip if not extracting all and not preferred language
                if not extract_all and language not in self.preferred_languages:
                    continue

                # Extract subtitle stream
                output_file = self.cache_dir / f"temp_{stream_index}.srt" if self.cache_dir else Path(f"/tmp/temp_{stream_index}.srt")

                try:
                    (
                        ffmpeg
                        .input(str(video_path))
                        .output(str(output_file), map=f"0:{stream_index}", codec="srt")
                        .overwrite_output()
                        .run(quiet=True)
                    )

                    # Parse extracted subtitles
                    segments = self._parse_subtitle_file(output_file, "srt")
                    if segments:
                        track = self._create_subtitle_track(segments, language, "srt", True, False)
                        tracks.append(track)

                    # Clean up temp file
                    if output_file.exists():
                        output_file.unlink()

                except Exception as e:
                    logger.warning(f"Failed to extract stream {stream_index}: {e}")

        except Exception as e:
            logger.error(f"Failed to probe video: {e}")

        return tracks

    def _find_external_subtitles(self, video_path: Path) -> List[SubtitleTrack]:
        """Find external subtitle files"""
        tracks = []
        video_stem = video_path.stem

        # Common subtitle extensions
        subtitle_extensions = [".srt", ".vtt", ".ass", ".ssa", ".sub"]

        for ext in subtitle_extensions:
            # Look for exact match
            subtitle_file = video_path.with_suffix(ext)
            if subtitle_file.exists():
                segments = self._parse_subtitle_file(subtitle_file, ext[1:])
                if segments:
                    track = self._create_subtitle_track(segments, "und", ext[1:], False, False)
                    tracks.append(track)

            # Look for language-specific files
            for lang in self.preferred_languages:
                patterns = [
                    video_path.parent / f"{video_stem}.{lang}{ext}",
                    video_path.parent / f"{video_stem}_{lang}{ext}",
                    video_path.parent / f"{video_stem}-{lang}{ext}"
                ]

                for pattern in patterns:
                    if pattern.exists():
                        segments = self._parse_subtitle_file(pattern, ext[1:])
                        if segments:
                            track = self._create_subtitle_track(segments, lang, ext[1:], False, False)
                            tracks.append(track)

        return tracks

    def _parse_subtitle_file(self, file_path: Path, format: str) -> List[SubtitleSegment]:
        """Parse subtitle file into segments"""
        segments = []

        try:
            if format == "srt":
                with open(file_path, "r", encoding="utf-8") as f:
                    subs = list(srt.parse(f.read()))
                    for sub in subs:
                        segment = SubtitleSegment(
                            text=sub.content,
                            start_time=sub.start.total_seconds(),
                            end_time=sub.end.total_seconds(),
                            confidence=1.0,
                            language="und"
                        )
                        segments.append(segment)

            elif format == "vtt":
                vtt = webvtt.read(str(file_path))
                for caption in vtt:
                    segment = SubtitleSegment(
                        text=caption.text,
                        start_time=self._vtt_time_to_seconds(caption.start),
                        end_time=self._vtt_time_to_seconds(caption.end),
                        confidence=1.0,
                        language="und"
                    )
                    segments.append(segment)

            elif format in ["ass", "ssa"]:
                subs = pysubs2.load(str(file_path))
                for line in subs:
                    segment = SubtitleSegment(
                        text=line.text,
                        start_time=line.start / 1000.0,
                        end_time=line.end / 1000.0,
                        confidence=1.0,
                        language="und"
                    )
                    segments.append(segment)

        except Exception as e:
            logger.error(f"Failed to parse {format} file: {e}")

        return segments

    def _vtt_time_to_seconds(self, time_str: str) -> float:
        """Convert WebVTT time string to seconds"""
        parts = time_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        else:
            return float(time_str)

    def _extract_ocr_subtitles(self, video_path: Path,
                              sample_rate: int = 1) -> Optional[SubtitleTrack]:
        """Extract hardcoded subtitles using OCR"""
        segments = []
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames for OCR
        frame_interval = int(fps * sample_rate)  # Sample every second by default
        current_text = ""
        current_start = 0

        for frame_num in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                break

            # Extract subtitle region (bottom 20% of frame)
            height, width = frame.shape[:2]
            subtitle_region = frame[int(height * 0.8):, :]

            # Preprocess for OCR
            gray = cv2.cvtColor(subtitle_region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            # OCR
            try:
                data = pytesseract.image_to_data(
                    binary,
                    output_type=pytesseract.Output.DICT,
                    config="--psm 6"  # Uniform text block
                )

                # Extract text with confidence
                text_parts = []
                confidences = []

                for i, conf in enumerate(data["conf"]):
                    if int(conf) > self.ocr_confidence_threshold * 100:
                        text = data["text"][i].strip()
                        if text:
                            text_parts.append(text)
                            confidences.append(float(conf) / 100)

                frame_text = " ".join(text_parts)
                avg_confidence = np.mean(confidences) if confidences else 0

                # Check if text changed
                if frame_text != current_text:
                    if current_text:
                        # Save previous segment
                        segment = SubtitleSegment(
                            text=current_text,
                            start_time=current_start,
                            end_time=frame_num / fps,
                            confidence=avg_confidence,
                            language="und"
                        )
                        segments.append(segment)

                    current_text = frame_text
                    current_start = frame_num / fps

            except Exception as e:
                logger.warning(f"OCR failed for frame {frame_num}: {e}")

        cap.release()

        if segments:
            return self._create_subtitle_track(segments, "und", "ocr", False, True)
        return None

    def _create_subtitle_track(self, segments: List[SubtitleSegment],
                              language: str,
                              format: str,
                              is_embedded: bool,
                              is_auto_generated: bool) -> SubtitleTrack:
        """Create subtitle track from segments"""
        if not segments:
            return None

        # Calculate statistics
        total_duration = segments[-1].end_time if segments else 0
        word_count = sum(len(seg.text.split()) for seg in segments)
        avg_reading_speed = (word_count / (total_duration / 60)) if total_duration > 0 else 0

        return SubtitleTrack(
            segments=segments,
            language=language,
            format=format,
            is_embedded=is_embedded,
            is_auto_generated=is_auto_generated,
            total_duration=total_duration,
            word_count=word_count,
            avg_reading_speed=avg_reading_speed
        )

    def _sort_by_language_preference(self, tracks: List[SubtitleTrack]) -> List[SubtitleTrack]:
        """Sort tracks by language preference"""

        def get_priority(track):
            if track.language in self.preferred_languages:
                return self.preferred_languages.index(track.language)
            return len(self.preferred_languages)

        return sorted(tracks, key=get_priority)

    def align_with_audio(self, subtitle_track: SubtitleTrack,
                        audio_segments: List[Dict[str, Any]],
                        max_offset: float = 2.0) -> SubtitleTrack:
        """
        Align subtitles with audio segments

        Args:
            subtitle_track: Subtitle track to align
            audio_segments: Audio segments with timing
            max_offset: Maximum allowed offset in seconds

        Returns:
            Aligned subtitle track
        """
        aligned_segments = []

        for sub_seg in subtitle_track.segments:
            best_match = None
            best_score = float("inf")

            for audio_seg in audio_segments:
                # Calculate overlap
                overlap_start = max(sub_seg.start_time, audio_seg["start"])
                overlap_end = min(sub_seg.end_time, audio_seg["end"])
                overlap = max(0, overlap_end - overlap_start)

                # Calculate offset
                offset = abs(sub_seg.start_time - audio_seg["start"])

                if offset < max_offset and overlap > 0:
                    score = offset - overlap  # Lower is better
                    if score < best_score:
                        best_score = score
                        best_match = audio_seg

            if best_match:
                # Create aligned segment
                aligned_seg = SubtitleSegment(
                    text=sub_seg.text,
                    start_time=best_match["start"],
                    end_time=best_match["end"],
                    confidence=sub_seg.confidence,
                    language=sub_seg.language,
                    speaker=best_match.get("speaker")
                )
                aligned_segments.append(aligned_seg)
            else:
                # Keep original if no match found
                aligned_segments.append(sub_seg)

        # Create new aligned track
        return self._create_subtitle_track(
            aligned_segments,
            subtitle_track.language,
            subtitle_track.format,
            subtitle_track.is_embedded,
            subtitle_track.is_auto_generated
        )