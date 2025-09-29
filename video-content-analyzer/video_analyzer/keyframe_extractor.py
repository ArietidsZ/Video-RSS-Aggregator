"""
Keyframe Extraction for Thumbnail Generation
Extracts visually representative frames from videos
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import time
from enum import Enum

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """Keyframe extraction methods"""

    UNIFORM = "uniform"  # Extract at uniform intervals
    SCENE_BASED = "scene_based"  # Extract from scene changes
    CONTENT_BASED = "content_based"  # Extract based on visual content
    MOTION_BASED = "motion_based"  # Extract based on motion analysis
    ML_BASED = "ml_based"  # ML-based extraction


@dataclass
class Keyframe:
    """Represents a keyframe"""

    frame: np.ndarray  # Frame image
    timestamp: float  # Timestamp in seconds
    frame_number: int  # Frame number
    score: float  # Quality/importance score
    metadata: Dict[str, Any]  # Additional metadata

    def to_pil(self) -> Image.Image:
        """Convert to PIL Image"""
        return Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

    def save(self, path: str):
        """Save keyframe to file"""
        cv2.imwrite(path, self.frame)


@dataclass
class KeyframeSet:
    """Collection of keyframes"""

    keyframes: List[Keyframe]
    video_duration: float
    video_fps: float
    extraction_method: str

    @property
    def best_keyframe(self) -> Optional[Keyframe]:
        """Get highest scoring keyframe"""
        if not self.keyframes:
            return None
        return max(self.keyframes, key=lambda k: k.score)

    def get_thumbnail_candidates(self, num: int = 3) -> List[Keyframe]:
        """Get top N thumbnail candidates"""
        sorted_frames = sorted(self.keyframes, key=lambda k: k.score, reverse=True)
        return sorted_frames[:num]

    def to_grid(self, cols: int = 3) -> np.ndarray:
        """Arrange keyframes in a grid"""
        if not self.keyframes:
            return np.array([])

        rows = (len(self.keyframes) + cols - 1) // cols
        frame_h, frame_w = self.keyframes[0].frame.shape[:2]

        # Create grid
        grid = np.zeros((rows * frame_h, cols * frame_w, 3), dtype=np.uint8)

        for i, keyframe in enumerate(self.keyframes):
            row = i // cols
            col = i % cols
            y1 = row * frame_h
            y2 = (row + 1) * frame_h
            x1 = col * frame_w
            x2 = (col + 1) * frame_w
            grid[y1:y2, x1:x2] = keyframe.frame

        return grid


class KeyframeExtractor:
    """
    Extract representative keyframes from videos
    Used for thumbnail generation and visual summaries
    """

    def __init__(self,
                 method: ExtractionMethod = ExtractionMethod.CONTENT_BASED,
                 num_keyframes: int = 10,
                 min_similarity_threshold: float = 0.7,
                 use_gpu: bool = True,
                 target_size: Tuple[int, int] = (640, 480)):
        """
        Initialize keyframe extractor

        Args:
            method: Extraction method to use
            num_keyframes: Target number of keyframes
            min_similarity_threshold: Minimum similarity to consider frames different
            use_gpu: Use GPU acceleration
            target_size: Target size for keyframes
        """
        self.method = method
        self.num_keyframes = num_keyframes
        self.min_similarity_threshold = min_similarity_threshold
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.target_size = target_size

        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        # Initialize models for ML-based extraction
        if method == ExtractionMethod.ML_BASED:
            self._init_ml_models()

        logger.info(f"KeyframeExtractor initialized with {method.value} method")

    def _init_ml_models(self):
        """Initialize ML models for keyframe extraction"""
        # Use MobileNet for efficiency
        self.feature_model = models.mobilenet_v3_small(pretrained=True)
        self.feature_model.classifier = torch.nn.Identity()
        self.feature_model = self.feature_model.to(self.device)
        self.feature_model.eval()

        # Aesthetic scorer (simple CNN)
        self.aesthetic_model = self._create_aesthetic_model()
        self.aesthetic_model = self.aesthetic_model.to(self.device)
        self.aesthetic_model.eval()

    def _create_aesthetic_model(self) -> torch.nn.Module:
        """Create simple aesthetic scoring model"""

        class AestheticNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(64, 1)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv2(x))
                x = self.pool(x)
                x = x.flatten(1)
                x = torch.sigmoid(self.fc(x))
                return x

        return AestheticNet()

    def extract(self, video_path: str,
               scenes: Optional[List[Tuple[float, float]]] = None) -> KeyframeSet:
        """
        Extract keyframes from video

        Args:
            video_path: Path to video file
            scenes: Optional scene boundaries for scene-based extraction

        Returns:
            Set of extracted keyframes
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        start_time = time.time()

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Extract keyframes based on method
        if self.method == ExtractionMethod.UNIFORM:
            keyframes = self._extract_uniform(cap, fps, frame_count)
        elif self.method == ExtractionMethod.SCENE_BASED:
            keyframes = self._extract_scene_based(cap, fps, scenes)
        elif self.method == ExtractionMethod.CONTENT_BASED:
            keyframes = self._extract_content_based(cap, fps, frame_count)
        elif self.method == ExtractionMethod.MOTION_BASED:
            keyframes = self._extract_motion_based(cap, fps, frame_count)
        elif self.method == ExtractionMethod.ML_BASED:
            keyframes = self._extract_ml_based(cap, fps, frame_count)
        else:
            keyframes = []

        cap.release()

        processing_time = time.time() - start_time
        logger.info(f"Extracted {len(keyframes)} keyframes in {processing_time:.2f}s")

        return KeyframeSet(
            keyframes=keyframes,
            video_duration=duration,
            video_fps=fps,
            extraction_method=self.method.value
        )

    def _extract_uniform(self, cap: cv2.VideoCapture,
                        fps: float,
                        frame_count: int) -> List[Keyframe]:
        """Extract keyframes at uniform intervals"""
        keyframes = []
        interval = max(1, frame_count // self.num_keyframes)

        for i in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                # Resize frame
                frame = cv2.resize(frame, self.target_size)

                keyframe = Keyframe(
                    frame=frame,
                    timestamp=i / fps,
                    frame_number=i,
                    score=1.0,  # Uniform extraction has equal scores
                    metadata={"method": "uniform"}
                )
                keyframes.append(keyframe)

                if len(keyframes) >= self.num_keyframes:
                    break

        return keyframes

    def _extract_scene_based(self, cap: cv2.VideoCapture,
                            fps: float,
                            scenes: Optional[List[Tuple[float, float]]]) -> List[Keyframe]:
        """Extract keyframes from scene changes"""
        if not scenes:
            # Fallback to uniform if no scenes provided
            return self._extract_uniform(cap, fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        keyframes = []

        for start_time, end_time in scenes:
            # Get middle frame of each scene
            middle_time = (start_time + end_time) / 2
            frame_num = int(middle_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, self.target_size)

                keyframe = Keyframe(
                    frame=frame,
                    timestamp=middle_time,
                    frame_number=frame_num,
                    score=1.0,
                    metadata={"method": "scene_based", "scene_duration": end_time - start_time}
                )
                keyframes.append(keyframe)

        # Sort by score and limit to num_keyframes
        keyframes = sorted(keyframes, key=lambda k: k.score, reverse=True)[:self.num_keyframes]
        return keyframes

    def _extract_content_based(self, cap: cv2.VideoCapture,
                              fps: float,
                              frame_count: int) -> List[Keyframe]:
        """Extract keyframes based on visual content diversity"""
        # Sample frames
        sample_interval = max(1, frame_count // (self.num_keyframes * 3))
        sampled_frames = []
        frame_numbers = []

        for i in range(0, frame_count, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, (224, 224))  # Smaller size for feature extraction
                sampled_frames.append(frame)
                frame_numbers.append(i)

        if not sampled_frames:
            return []

        # Extract features
        features = self._extract_visual_features(sampled_frames)

        # Cluster frames
        if len(features) > self.num_keyframes:
            kmeans = KMeans(n_clusters=self.num_keyframes, random_state=42)
            labels = kmeans.fit_predict(features)

            # Get representative frame from each cluster
            keyframes = []
            for cluster_id in range(self.num_keyframes):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    # Get frame closest to cluster center
                    cluster_features = features[cluster_indices]
                    center = kmeans.cluster_centers_[cluster_id]
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    best_idx = cluster_indices[np.argmin(distances)]

                    # Get full resolution frame
                    frame_num = frame_numbers[best_idx]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()

                    if ret:
                        frame = cv2.resize(frame, self.target_size)
                        keyframe = Keyframe(
                            frame=frame,
                            timestamp=frame_num / fps,
                            frame_number=frame_num,
                            score=1.0 / (1 + distances.min()),  # Higher score for frames closer to center
                            metadata={"method": "content_based", "cluster": int(cluster_id)}
                        )
                        keyframes.append(keyframe)
        else:
            # If not enough frames, use all
            keyframes = []
            for i, frame in enumerate(sampled_frames):
                frame = cv2.resize(frame, self.target_size)
                keyframe = Keyframe(
                    frame=frame,
                    timestamp=frame_numbers[i] / fps,
                    frame_number=frame_numbers[i],
                    score=1.0,
                    metadata={"method": "content_based"}
                )
                keyframes.append(keyframe)

        return keyframes

    def _extract_motion_based(self, cap: cv2.VideoCapture,
                             fps: float,
                             frame_count: int) -> List[Keyframe]:
        """Extract keyframes based on motion analysis"""
        keyframes = []
        prev_frame = None
        motion_scores = []
        sample_interval = max(1, fps)  # Sample once per second

        # Calculate motion scores
        for i in range(0, frame_count, int(sample_interval)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 240))  # Smaller size for motion detection

            if prev_frame is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

                # Calculate motion magnitude
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_score = np.mean(magnitude)
                motion_scores.append((i, motion_score))

            prev_frame = gray

        if not motion_scores:
            return self._extract_uniform(cap, fps, frame_count)

        # Select frames with significant motion changes
        motion_scores.sort(key=lambda x: x[1], reverse=True)

        for frame_num, score in motion_scores[:self.num_keyframes]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                frame = cv2.resize(frame, self.target_size)
                keyframe = Keyframe(
                    frame=frame,
                    timestamp=frame_num / fps,
                    frame_number=frame_num,
                    score=score,
                    metadata={"method": "motion_based", "motion_score": float(score)}
                )
                keyframes.append(keyframe)

        return sorted(keyframes, key=lambda k: k.timestamp)

    def _extract_ml_based(self, cap: cv2.VideoCapture,
                         fps: float,
                         frame_count: int) -> List[Keyframe]:
        """Extract keyframes using ML models"""
        if not hasattr(self, "feature_model"):
            return self._extract_content_based(cap, fps, frame_count)

        # Sample frames
        sample_interval = max(1, frame_count // (self.num_keyframes * 3))
        candidates = []

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for i in range(0, frame_count, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                continue

            # Calculate aesthetic score
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Get features
                features = self.feature_model(tensor)

                # Get aesthetic score
                if hasattr(self, "aesthetic_model"):
                    score = self.aesthetic_model(tensor).item()
                else:
                    # Simple quality metrics
                    score = self._calculate_quality_score(frame)

            candidates.append({
                "frame_num": i,
                "frame": frame,
                "features": features.cpu().numpy().flatten(),
                "score": score
            })

        # Select diverse high-quality frames
        selected = self._select_diverse_frames(candidates, self.num_keyframes)

        keyframes = []
        for item in selected:
            frame = cv2.resize(item["frame"], self.target_size)
            keyframe = Keyframe(
                frame=frame,
                timestamp=item["frame_num"] / fps,
                frame_number=item["frame_num"],
                score=item["score"],
                metadata={"method": "ml_based", "aesthetic_score": item["score"]}
            )
            keyframes.append(keyframe)

        return sorted(keyframes, key=lambda k: k.timestamp)

    def _extract_visual_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """Extract visual features from frames"""
        features = []

        for frame in frames:
            # Color histogram
            hist_b = cv2.calcHist([frame], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([frame], [2], None, [32], [0, 256])
            color_features = np.concatenate([hist_b, hist_g, hist_r]).flatten()

            # Edge features
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Texture features (simplified)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture = np.var(laplacian)

            # Combine features
            frame_features = np.concatenate([
                color_features / np.sum(color_features),  # Normalize
                [edge_density, texture]
            ])
            features.append(frame_features)

        return np.array(features)

    def _calculate_quality_score(self, frame: np.ndarray) -> float:
        """Calculate frame quality score"""
        # Sharpness (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)

        # Contrast
        contrast = gray.std()

        # Colorfulness
        b, g, r = cv2.split(frame)
        rg = np.abs(r.astype(float) - g.astype(float))
        yb = np.abs(0.5 * (r.astype(float) + g.astype(float)) - b.astype(float))
        colorfulness = np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)

        # Combine scores
        score = (
            0.4 * min(sharpness / 1000, 1.0) +  # Normalized sharpness
            0.3 * min(contrast / 50, 1.0) +  # Normalized contrast
            0.3 * min(colorfulness / 100, 1.0)  # Normalized colorfulness
        )

        return score

    def _select_diverse_frames(self, candidates: List[Dict],
                              num_frames: int) -> List[Dict]:
        """Select diverse high-quality frames"""
        if len(candidates) <= num_frames:
            return candidates

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Select top frame
        selected = [candidates[0]]
        remaining = candidates[1:]

        # Select diverse frames
        while len(selected) < num_frames and remaining:
            max_min_distance = -1
            best_candidate = None
            best_idx = -1

            for i, candidate in enumerate(remaining):
                # Calculate minimum distance to already selected frames
                min_distance = float("inf")
                for sel in selected:
                    distance = np.linalg.norm(
                        candidate["features"] - sel["features"]
                    )
                    min_distance = min(min_distance, distance)

                # Weighted score combining quality and diversity
                weighted_score = candidate["score"] * 0.3 + min_distance * 0.7

                if weighted_score > max_min_distance:
                    max_min_distance = weighted_score
                    best_candidate = candidate
                    best_idx = i

            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break

        return selected