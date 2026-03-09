"""Liveness detection to distinguish a real face from a photo or screen spoof.

Detection methods used:
1. Eye-blink detection  – photos and static screens cannot blink.
2. Micro-movement analysis – a live face produces small natural movements.
3. Texture analysis – skin has characteristic high-frequency texture; flat
   printed photos or screen displays produce lower Laplacian variance.
"""

import time
from collections import deque

import cv2
import numpy as np


class LivenessDetector:
    """Detects whether the face belongs to a live person or a spoof attempt.

    Args:
        blink_window: Time window (seconds) in which blinks are counted.
        required_blinks: Minimum blinks required within ``blink_window`` to
            be considered live.
    """

    def __init__(self, blink_window: float = 10.0, required_blinks: int = 1):
        self.blink_window = blink_window
        self.required_blinks = required_blinks

        # Eye presence history (True = eyes detected)
        self._eye_history: deque = deque(maxlen=10)
        self._blink_times: deque = deque()
        self._in_blink: bool = False

        # Face-centroid positions for micro-movement analysis
        self._face_positions: deque = deque(maxlen=30)

        # Laplacian texture scores
        self._texture_scores: deque = deque(maxlen=30)

        # Public state
        self.is_live: bool = False
        self.liveness_score: float = 0.0
        self.status_message: str = "Checking liveness…"

        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, frame, face_roi, eyes):
        """Update the detector with a new video frame.

        Args:
            frame: Full BGR image array.
            face_roi: (x, y, w, h) tuple, or None if no face detected.
            eyes: Array of eye detections from the face region, or None.

        Returns:
            Tuple (is_live: bool, liveness_score: float).
        """
        current_time = time.time()

        eyes_detected = (eyes is not None) and (len(eyes) >= 2)
        self._detect_blink(eyes_detected, current_time)

        if face_roi is not None:
            x, y, w, h = face_roi
            self._face_positions.append((x + w / 2.0, y + h / 2.0))
            if frame is not None:
                self._texture_scores.append(self._analyze_texture(frame, face_roi))

        self._compute_liveness_score(current_time)
        return self.is_live, self.liveness_score

    def get_blink_count(self) -> int:
        """Return the number of blinks recorded in the recent window."""
        return len(self._blink_times)

    def reset(self):
        """Reset all internal state."""
        self._eye_history.clear()
        self._blink_times.clear()
        self._face_positions.clear()
        self._texture_scores.clear()
        self._in_blink = False
        self.is_live = False
        self.liveness_score = 0.0
        self.status_message = "Checking liveness…"
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_blink(self, eyes_detected: bool, current_time: float):
        """Update blink tracking from eye-presence signal."""
        self._eye_history.append(eyes_detected)

        if len(self._eye_history) >= 3:
            # Falling edge: eyes open → closed
            if self._eye_history[-2] and not self._eye_history[-1]:
                self._in_blink = True
            # Rising edge after blink: eyes closed → open (completed blink)
            elif self._in_blink and self._eye_history[-1]:
                self._blink_times.append(current_time)
                self._in_blink = False

        # Expire old blink records
        cutoff = current_time - self.blink_window
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

    def _analyze_texture(self, frame, face_roi) -> float:
        """Compute a texture score in [0, 1] for the face region.

        Real faces exhibit rich high-frequency texture (skin pores, hair).
        Printed photos or screens have lower Laplacian variance.
        """
        x, y, w, h = face_roi
        face_region = frame[y : y + h, x : x + w]

        if face_region.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Normalize: 500 empirically corresponds to typical live-face sharpness
        return float(min(1.0, lap_var / 500.0))

    def _compute_liveness_score(self, current_time: float):
        """Aggregate individual cues into an overall liveness score."""
        elapsed = current_time - self._start_time

        # --- Blink cue (50 % weight) ---
        recent_blinks = len(self._blink_times)
        if elapsed < self.blink_window:
            blink_score = min(1.0, recent_blinks / max(1, self.required_blinks))
        else:
            blink_score = 1.0 if recent_blinks >= self.required_blinks else 0.0

        # --- Micro-movement cue (20 % weight) ---
        if len(self._face_positions) >= 5:
            positions = np.array(list(self._face_positions))
            movement = float(np.std(positions, axis=0).mean())
            movement_score = min(1.0, movement / 5.0)  # 5 px → full score
        else:
            movement_score = 0.1  # conservative default

        # --- Texture cue (30 % weight) ---
        if len(self._texture_scores) >= 5:
            texture_score = float(np.mean(list(self._texture_scores)))
        else:
            texture_score = 0.15  # conservative default

        self.liveness_score = (
            blink_score * 0.5 + movement_score * 0.2 + texture_score * 0.3
        )

        if elapsed < 3.0:
            self.is_live = False
            self.status_message = "Analyzing… (please wait)"
        elif self.liveness_score >= 0.5:
            self.is_live = True
            self.status_message = f"Live – score: {self.liveness_score:.2f}"
        else:
            self.is_live = False
            self.status_message = f"Spoof detected – score: {self.liveness_score:.2f}"
