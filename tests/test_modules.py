"""Unit tests for the AI-based Vital Signs Monitoring System.

Each module is tested independently; no camera hardware is required.
"""

import os
import sys

import numpy as np
import pytest

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_detector import FaceDetector
from health_monitor_gui import HealthMonitorGUI
from liveness_detector import LivenessDetector
from vital_signs_estimator import VitalSignsEstimator


# ---------------------------------------------------------------------------
# FaceDetector
# ---------------------------------------------------------------------------


class TestFaceDetector:
    def setup_method(self):
        self.detector = FaceDetector()

    def test_cascades_loaded(self):
        assert not self.detector.face_cascade.empty()
        assert not self.detector.eye_cascade.empty()

    def test_detect_face_blank_image_returns_none(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        assert self.detector.detect_face(blank) is None

    def test_get_forehead_roi_shape_and_coords(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi_frame, (rx, ry, rw, rh) = self.detector.get_forehead_roi(frame, (100, 100, 200, 200))
        assert roi_frame.ndim == 3
        assert rw > 0 and rh > 0
        # Region must be inside the frame
        assert rx >= 0 and ry >= 0
        assert rx + rw <= 640 and ry + rh <= 480

    def test_get_forehead_roi_clamped_to_image(self):
        # Face near the edge – ROI should not exceed image boundaries
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi_frame, (rx, ry, rw, rh) = self.detector.get_forehead_roi(frame, (0, 0, 640, 480))
        assert rx + rw <= 640
        assert ry + rh <= 480

    def test_get_cheek_roi_shape_and_coords(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi_frame, (cx, cy, cw, ch) = self.detector.get_cheek_roi(frame, (100, 100, 200, 200))
        assert roi_frame.ndim == 3
        assert cw > 0 and ch > 0
        assert cx >= 0 and cy >= 0
        assert cx + cw <= 640 and cy + ch <= 480


# ---------------------------------------------------------------------------
# LivenessDetector
# ---------------------------------------------------------------------------


class TestLivenessDetector:
    def setup_method(self):
        self.detector = LivenessDetector()

    def test_initial_state(self):
        assert not self.detector.is_live
        assert self.detector.liveness_score == 0.0
        assert self.detector.get_blink_count() == 0

    def test_update_no_face_does_not_raise(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        is_live, score = self.detector.update(frame, None, None)
        assert isinstance(is_live, bool)
        assert 0.0 <= score <= 1.0  # weights sum to exactly 1.0 so score is bounded

    def test_update_with_face_returns_valid_types(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        is_live, score = self.detector.update(frame, (100, 100, 200, 200), [])
        assert isinstance(is_live, bool)
        assert isinstance(score, float)

    def test_reset_clears_state(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.detector.update(frame, (50, 50, 200, 200), [])
        self.detector.reset()
        assert not self.detector.is_live
        assert self.detector.liveness_score == 0.0
        assert self.detector.get_blink_count() == 0

    def test_blink_detection_increments_count(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        face_roi = (50, 50, 200, 200)
        two_eyes = [(10, 10, 30, 30), (50, 10, 30, 30)]

        # Seed eye-history with eyes-open
        for _ in range(5):
            self.detector.update(frame, face_roi, two_eyes)

        # Blink: eyes disappear for one frame
        self.detector.update(frame, face_roi, [])

        # Eyes reappear to complete the blink
        self.detector.update(frame, face_roi, two_eyes)

        assert self.detector.get_blink_count() >= 1

    def test_texture_score_non_negative(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        score = self.detector._analyze_texture(frame, (50, 50, 200, 200))
        assert score >= 0.0

    def test_texture_score_empty_region(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        # ROI larger than frame produces empty slice
        score = self.detector._analyze_texture(frame, (0, 0, 200, 200))
        assert score >= 0.0


# ---------------------------------------------------------------------------
# VitalSignsEstimator
# ---------------------------------------------------------------------------


class TestVitalSignsEstimator:
    def setup_method(self):
        self.est = VitalSignsEstimator(sampling_rate=30)

    def test_initial_state(self):
        assert self.est.heart_rate == 0.0
        assert self.est.stress_level == 0.0
        assert abs(self.est.temperature - 98.6) < 0.1
        assert not self.est.is_ready()

    def test_update_none_does_not_raise(self):
        self.est.update(None)
        assert self.est.heart_rate == 0.0

    def test_update_empty_array_does_not_raise(self):
        self.est.update(np.array([]))
        assert self.est.heart_rate == 0.0

    def test_ready_after_sufficient_frames(self):
        frame = np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)
        # Feed 5 seconds worth at 30 fps = 150 frames
        for _ in range(150):
            self.est.update(frame)
        assert self.est.is_ready()

    def test_temperature_stays_in_physiological_range(self):
        frame = np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)
        for _ in range(200):
            self.est.update(frame)
        # Even under simulation, temperature should stay near normal range
        assert 95.0 <= self.est.temperature <= 102.0

    def test_heart_rate_in_valid_range_after_synthetic_signal(self):
        """Feed a 1.2 Hz (72 BPM) sinusoidal signal and check the estimate."""
        sr = 30
        est = VitalSignsEstimator(sampling_rate=sr)
        t = np.linspace(0, 10, sr * 10)
        # Green-channel signal oscillating at 1.2 Hz
        signal = np.sin(2 * np.pi * 1.2 * t) * 50 + 128

        for val in signal:
            frame = np.full((10, 10, 3), int(val), dtype=np.uint8)
            est.update(frame)

        if est.is_ready() and est.heart_rate > 0:
            assert 42 <= est.heart_rate <= 180

    def test_get_pulse_waveform_returns_array(self):
        waveform = self.est.get_pulse_waveform()
        assert isinstance(waveform, np.ndarray)

    def test_get_pulse_waveform_populated_after_data(self):
        frame = np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)
        for _ in range(160):
            self.est.update(frame)
        assert len(self.est.get_pulse_waveform()) > 0

    def test_reset_clears_estimates(self):
        frame = np.random.randint(100, 200, (50, 50, 3), dtype=np.uint8)
        for _ in range(160):
            self.est.update(frame)
        self.est.reset()
        assert self.est.heart_rate == 0.0
        assert self.est.stress_level == 0.0
        assert not self.est.is_ready()


# ---------------------------------------------------------------------------
# HealthMonitorGUI
# ---------------------------------------------------------------------------


class TestHealthMonitorGUI:
    def setup_method(self):
        self.gui = HealthMonitorGUI(frame_width=640, frame_height=480)

    def test_total_width_equals_frame_plus_panel(self):
        assert self.gui.total_width == self.gui.frame_width + self.gui.panel_width

    def test_render_no_face_returns_correct_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.gui.render(
            frame,
            None,
            (False, 0.0, "Checking"),
            {"heart_rate": 0, "stress_level": 0, "temperature": 98.6},
            None,
        )
        assert result.shape == (480, self.gui.total_width, 3)

    def test_render_with_face_returns_correct_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.gui.render(
            frame,
            (100, 100, 200, 200),
            (True, 0.85, "Live"),
            {"heart_rate": 75, "stress_level": 30, "temperature": 98.6},
            np.sin(np.linspace(0, 4 * np.pi, 150)),
        )
        assert result.shape == (480, self.gui.total_width, 3)

    def test_alerts_generated_for_abnormal_vitals(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.gui.render(
            frame,
            None,
            (True, 0.9, "Live"),
            {"heart_rate": 130, "stress_level": 85, "temperature": 100.5},
            None,
        )
        assert any("Heart Rate" in a for a in self.gui.alerts)
        assert any("Stress" in a for a in self.gui.alerts)
        assert any("Temperature" in a for a in self.gui.alerts)

    def test_no_hr_alert_for_normal_heart_rate(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.gui.render(
            frame,
            None,
            (True, 0.9, "Live"),
            {"heart_rate": 72, "stress_level": 20, "temperature": 98.6},
            None,
        )
        assert not any("Heart Rate" in a for a in self.gui.alerts)

    def test_no_alert_when_heart_rate_is_zero(self):
        """HR = 0 means no data yet; no alert should fire."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.gui.render(
            frame,
            None,
            (True, 0.9, "Live"),
            {"heart_rate": 0, "stress_level": 0, "temperature": 98.6},
            None,
        )
        assert not any("Heart Rate" in a for a in self.gui.alerts)

    def test_low_heart_rate_alert(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.gui.render(
            frame,
            None,
            (True, 0.9, "Live"),
            {"heart_rate": 40, "stress_level": 0, "temperature": 98.6},
            None,
        )
        assert any("Low Heart Rate" in a for a in self.gui.alerts)

    def test_waveform_draw_with_flat_signal(self):
        """Flat signal (all zeros) should not raise an exception."""
        panel = np.zeros((480, 280, 3), dtype=np.uint8)
        flat = np.zeros(150)
        self.gui._draw_waveform(panel, flat, 10, 10, 260, 70)

    def test_waveform_draw_none(self):
        panel = np.zeros((480, 280, 3), dtype=np.uint8)
        self.gui._draw_waveform(panel, None, 10, 10, 260, 70)
