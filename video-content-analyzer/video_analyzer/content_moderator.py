"""
Content Moderation Filters
Detects NSFW, violence, and other inappropriate content
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import time
from enum import Enum

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline, AutoModelForImageClassification, AutoProcessor
from ultralytics import YOLO
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content to moderate"""

    SAFE = "safe"
    NSFW = "nsfw"
    VIOLENCE = "violence"
    GORE = "gore"
    HATE = "hate"
    DRUGS = "drugs"
    WEAPONS = "weapons"
    GAMBLING = "gambling"
    TOBACCO = "tobacco"
    ALCOHOL = "alcohol"


@dataclass
class ModerationFlag:
    """Single moderation flag"""

    content_type: ContentType
    confidence: float
    timestamp: float
    frame_number: int
    description: str
    severity: str  # low, medium, high

    def is_severe(self) -> bool:
        return self.severity == "high"


@dataclass
class ModerationResult:
    """Complete moderation result"""

    is_safe: bool
    overall_safety_score: float
    flags: List[ModerationFlag]
    frame_analysis_count: int
    text_analysis_count: int
    audio_analysis_count: int
    processing_time: float
    content_rating: str  # G, PG, PG-13, R, NC-17
    moderation_summary: str

    def get_flagged_timestamps(self) -> List[float]:
        """Get timestamps of flagged content"""
        return [flag.timestamp for flag in self.flags]

    def get_severity_distribution(self) -> Dict[str, int]:
        """Get distribution of flag severities"""
        distribution = {"low": 0, "medium": 0, "high": 0}
        for flag in self.flags:
            distribution[flag.severity] += 1
        return distribution


class ContentModerator:
    """
    Multi-modal content moderation
    Analyzes video, audio, and text for inappropriate content
    """

    def __init__(self,
                 sensitivity: str = "medium",
                 content_types: Optional[List[ContentType]] = None,
                 use_gpu: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialize content moderator

        Args:
            sensitivity: Detection sensitivity ("low", "medium", "high")
            content_types: Content types to detect (None for all)
            use_gpu: Use GPU acceleration
            cache_dir: Directory for model cache
        """
        self.sensitivity = sensitivity
        self.content_types = content_types or list(ContentType)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # Set thresholds based on sensitivity
        self._set_thresholds()

        # Initialize models
        self._init_models()

        logger.info(f"ContentModerator initialized with {sensitivity} sensitivity")

    def _set_thresholds(self):
        """Set detection thresholds based on sensitivity"""
        if self.sensitivity == "low":
            self.nsfw_threshold = 0.8
            self.violence_threshold = 0.8
            self.general_threshold = 0.7
        elif self.sensitivity == "medium":
            self.nsfw_threshold = 0.6
            self.violence_threshold = 0.6
            self.general_threshold = 0.5
        else:  # high
            self.nsfw_threshold = 0.4
            self.violence_threshold = 0.4
            self.general_threshold = 0.3

    def _init_models(self):
        """Initialize moderation models"""
        # NSFW detection model
        try:
            self.nsfw_model = self._load_nsfw_model()
        except Exception as e:
            logger.warning(f"Failed to load NSFW model: {e}")
            self.nsfw_model = None

        # Violence detection model
        try:
            self.violence_model = self._load_violence_model()
        except Exception as e:
            logger.warning(f"Failed to load violence model: {e}")
            self.violence_model = None

        # Object detection for weapons, drugs, etc.
        try:
            self.object_detector = YOLO("yolov8m.pt")
            if self.use_gpu:
                self.object_detector.to(self.device)
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}")
            self.object_detector = None

        # Text toxicity classifier
        try:
            self.text_classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if self.use_gpu else -1
            )
        except Exception as e:
            logger.warning(f"Failed to load text classifier: {e}")
            self.text_classifier = None

    def _load_nsfw_model(self):
        """Load NSFW detection model"""
        # Use a lightweight ONNX model for NSFW detection
        model_path = self.cache_dir / "nsfw_model.onnx" if self.cache_dir else "nsfw_model.onnx"

        if not model_path.exists():
            # Download or use a pretrained model
            # For demo purposes, we'll use a simple CNN
            return self._create_simple_classifier("nsfw")

        # Load ONNX model
        providers = ["CUDAExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]
        return ort.InferenceSession(str(model_path), providers=providers)

    def _load_violence_model(self):
        """Load violence detection model"""
        return self._create_simple_classifier("violence")

    def _create_simple_classifier(self, task: str):
        """Create a simple CNN classifier for demo"""

        class SimpleClassifier(torch.nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(32, 64, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(2),
                    torch.nn.Conv2d(64, 128, 3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = torch.nn.Linear(128, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return F.softmax(x, dim=1)

        model = SimpleClassifier()
        model = model.to(self.device)
        model.eval()
        return model

    def moderate(self, video_path: str,
                sample_rate: int = 1,
                transcript: Optional[List[Dict[str, Any]]] = None,
                analyze_audio: bool = True) -> ModerationResult:
        """
        Perform content moderation on video

        Args:
            video_path: Path to video file
            sample_rate: Frame sampling rate (analyze every Nth frame)
            transcript: Optional transcript for text analysis
            analyze_audio: Analyze audio content

        Returns:
            Moderation result
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        start_time = time.time()
        flags = []

        # Analyze video frames
        frame_flags, frame_count = self._analyze_video_frames(video_path, sample_rate)
        flags.extend(frame_flags)

        # Analyze text if transcript provided
        text_count = 0
        if transcript:
            text_flags, text_count = self._analyze_text(transcript)
            flags.extend(text_flags)

        # Analyze audio if requested
        audio_count = 0
        if analyze_audio:
            audio_flags, audio_count = self._analyze_audio(video_path)
            flags.extend(audio_flags)

        # Calculate overall safety
        is_safe, safety_score = self._calculate_safety(flags)

        # Determine content rating
        content_rating = self._determine_rating(flags)

        # Generate summary
        summary = self._generate_summary(flags, is_safe)

        processing_time = time.time() - start_time

        logger.info(
            f"Moderation complete: {'SAFE' if is_safe else 'FLAGGED'} "
            f"({len(flags)} flags) in {processing_time:.2f}s"
        )

        return ModerationResult(
            is_safe=is_safe,
            overall_safety_score=safety_score,
            flags=flags,
            frame_analysis_count=frame_count,
            text_analysis_count=text_count,
            audio_analysis_count=audio_count,
            processing_time=processing_time,
            content_rating=content_rating,
            moderation_summary=summary
        )

    def _analyze_video_frames(self, video_path: Path,
                            sample_rate: int) -> Tuple[List[ModerationFlag], int]:
        """Analyze video frames for inappropriate content"""
        flags = []
        frame_count = 0

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames
        for frame_num in range(0, total_frames, sample_rate * int(fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            timestamp = frame_num / fps

            # Check NSFW content
            if ContentType.NSFW in self.content_types and self.nsfw_model:
                nsfw_score = self._check_nsfw(frame)
                if nsfw_score > self.nsfw_threshold:
                    flag = ModerationFlag(
                        content_type=ContentType.NSFW,
                        confidence=nsfw_score,
                        timestamp=timestamp,
                        frame_number=frame_num,
                        description="Potentially inappropriate content detected",
                        severity=self._get_severity(nsfw_score, self.nsfw_threshold)
                    )
                    flags.append(flag)

            # Check violence
            if ContentType.VIOLENCE in self.content_types and self.violence_model:
                violence_score = self._check_violence(frame)
                if violence_score > self.violence_threshold:
                    flag = ModerationFlag(
                        content_type=ContentType.VIOLENCE,
                        confidence=violence_score,
                        timestamp=timestamp,
                        frame_number=frame_num,
                        description="Violence detected",
                        severity=self._get_severity(violence_score, self.violence_threshold)
                    )
                    flags.append(flag)

            # Check for weapons, drugs, etc. using object detection
            if self.object_detector:
                object_flags = self._check_objects(frame, timestamp, frame_num)
                flags.extend(object_flags)

        cap.release()
        return flags, frame_count

    def _check_nsfw(self, frame: np.ndarray) -> float:
        """Check frame for NSFW content"""
        if not self.nsfw_model:
            return 0.0

        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_normalized = frame_resized.astype(np.float32) / 255.0

        # Convert to tensor
        tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            if isinstance(self.nsfw_model, torch.nn.Module):
                output = self.nsfw_model(tensor)
                nsfw_score = output[0, 1].item()  # Assuming binary classification
            else:
                # ONNX model
                nsfw_score = 0.0  # Placeholder

        return nsfw_score

    def _check_violence(self, frame: np.ndarray) -> float:
        """Check frame for violent content"""
        if not self.violence_model:
            return 0.0

        # Similar to NSFW check
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frame_normalized = frame_resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            if isinstance(self.violence_model, torch.nn.Module):
                output = self.violence_model(tensor)
                violence_score = output[0, 1].item()
            else:
                violence_score = 0.0

        return violence_score

    def _check_objects(self, frame: np.ndarray,
                      timestamp: float,
                      frame_number: int) -> List[ModerationFlag]:
        """Check for specific objects (weapons, drugs, etc.)"""
        if not self.object_detector:
            return []

        flags = []

        # Run YOLO detection
        results = self.object_detector(frame, verbose=False)

        # Map YOLO classes to content types
        dangerous_objects = {
            "knife": ContentType.WEAPONS,
            "gun": ContentType.WEAPONS,
            "cigarette": ContentType.TOBACCO,
            "bottle": ContentType.ALCOHOL,  # Could be alcohol
        }

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls)
                    class_name = self.object_detector.names[class_id]
                    confidence = float(box.conf)

                    if class_name.lower() in dangerous_objects and confidence > self.general_threshold:
                        content_type = dangerous_objects[class_name.lower()]
                        if content_type in self.content_types:
                            flag = ModerationFlag(
                                content_type=content_type,
                                confidence=confidence,
                                timestamp=timestamp,
                                frame_number=frame_number,
                                description=f"{class_name} detected",
                                severity=self._get_severity(confidence, self.general_threshold)
                            )
                            flags.append(flag)

        return flags

    def _analyze_text(self, transcript: List[Dict[str, Any]]) -> Tuple[List[ModerationFlag], int]:
        """Analyze transcript for inappropriate content"""
        if not self.text_classifier or not transcript:
            return [], 0

        flags = []
        text_count = 0

        for segment in transcript:
            text = segment.get("text", "")
            timestamp = segment.get("start_time", 0)

            if not text:
                continue

            text_count += 1

            # Check for toxicity
            try:
                results = self.text_classifier(text)
                for result in results:
                    if result["label"] == "TOXIC" and result["score"] > self.general_threshold:
                        flag = ModerationFlag(
                            content_type=ContentType.HATE,
                            confidence=result["score"],
                            timestamp=timestamp,
                            frame_number=0,
                            description=f"Potentially toxic text: {text[:100]}",
                            severity=self._get_severity(result["score"], self.general_threshold)
                        )
                        flags.append(flag)
            except Exception as e:
                logger.warning(f"Text analysis failed: {e}")

        return flags, text_count

    def _analyze_audio(self, video_path: Path) -> Tuple[List[ModerationFlag], int]:
        """Analyze audio for inappropriate content"""
        # Placeholder for audio analysis
        # Could include: profanity detection, aggression detection, etc.
        return [], 0

    def _calculate_safety(self, flags: List[ModerationFlag]) -> Tuple[bool, float]:
        """Calculate overall safety score"""
        if not flags:
            return True, 1.0

        # Check for severe flags
        severe_flags = [f for f in flags if f.is_severe()]
        if severe_flags:
            return False, 0.0

        # Calculate weighted score
        total_weight = 0
        weighted_sum = 0

        content_weights = {
            ContentType.NSFW: 2.0,
            ContentType.VIOLENCE: 2.0,
            ContentType.GORE: 3.0,
            ContentType.HATE: 2.0,
            ContentType.WEAPONS: 1.5,
            ContentType.DRUGS: 1.5,
            ContentType.ALCOHOL: 0.5,
            ContentType.TOBACCO: 0.5,
            ContentType.GAMBLING: 1.0
        }

        for flag in flags:
            weight = content_weights.get(flag.content_type, 1.0)
            total_weight += weight
            weighted_sum += weight * (1 - flag.confidence)

        safety_score = weighted_sum / total_weight if total_weight > 0 else 1.0

        # Determine if safe based on score
        is_safe = safety_score > 0.5

        return is_safe, safety_score

    def _determine_rating(self, flags: List[ModerationFlag]) -> str:
        """Determine content rating based on flags"""
        if not flags:
            return "G"

        # Check content types
        has_nsfw = any(f.content_type == ContentType.NSFW for f in flags)
        has_violence = any(f.content_type == ContentType.VIOLENCE for f in flags)
        has_gore = any(f.content_type == ContentType.GORE for f in flags)
        has_profanity = any(f.content_type == ContentType.HATE for f in flags)
        has_substances = any(f.content_type in [ContentType.DRUGS, ContentType.ALCOHOL, ContentType.TOBACCO] for f in flags)

        # Determine rating
        if has_gore or (has_nsfw and any(f.is_severe() for f in flags if f.content_type == ContentType.NSFW)):
            return "NC-17"
        elif has_nsfw or (has_violence and any(f.is_severe() for f in flags if f.content_type == ContentType.VIOLENCE)):
            return "R"
        elif has_violence or has_profanity or has_substances:
            return "PG-13"
        elif len(flags) > 0:
            return "PG"
        else:
            return "G"

    def _get_severity(self, score: float, threshold: float) -> str:
        """Determine severity based on score"""
        if score > threshold + 0.3:
            return "high"
        elif score > threshold + 0.1:
            return "medium"
        else:
            return "low"

    def _generate_summary(self, flags: List[ModerationFlag], is_safe: bool) -> str:
        """Generate moderation summary"""
        if not flags:
            return "Content appears to be safe with no flags detected."

        # Count flags by type
        flag_counts = {}
        for flag in flags:
            if flag.content_type not in flag_counts:
                flag_counts[flag.content_type] = 0
            flag_counts[flag.content_type] += 1

        # Generate summary
        summary_parts = []
        summary_parts.append(f"Content {'is considered safe' if is_safe else 'has been flagged'} with {len(flags)} total flags.")

        for content_type, count in flag_counts.items():
            summary_parts.append(f"- {content_type.value.capitalize()}: {count} instance(s)")

        severity_dist = self._get_severity_distribution(flags)
        summary_parts.append(f"Severity distribution: {severity_dist}")

        return "\n".join(summary_parts)

    def _get_severity_distribution(self, flags: List[ModerationFlag]) -> str:
        """Get severity distribution string"""
        dist = {"low": 0, "medium": 0, "high": 0}
        for flag in flags:
            dist[flag.severity] += 1

        return f"Low: {dist['low']}, Medium: {dist['medium']}, High: {dist['high']}"