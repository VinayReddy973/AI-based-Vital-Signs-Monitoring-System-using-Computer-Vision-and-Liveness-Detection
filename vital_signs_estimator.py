"""Vital signs estimation from facial video using remote photoplethysmography (rPPG).

The green channel of the face ROI contains a subtle periodic signal caused by
blood volume changes under the skin.  By bandpass-filtering this signal and
applying an FFT we can estimate heart rate.  Heart-rate variability (HRV)
metrics derived from the same signal are used to proxy stress level.
Temperature is simulated with a physiological model (a real deployment would
use a thermal camera).
"""

import time
from collections import deque

import numpy as np
import scipy.signal as sp_signal


class VitalSignsEstimator:
    """Estimates heart rate (BPM), stress level (%), and temperature (°F).

    Args:
        sampling_rate: Expected video frame rate in frames-per-second.
    """

    # Valid physiological heart-rate range
    _HR_MIN_HZ: float = 0.7   # 42 BPM
    _HR_MAX_HZ: float = 3.0   # 180 BPM

    def __init__(self, sampling_rate: int = 30):
        self.sampling_rate = sampling_rate
        self._window_size = sampling_rate * 10  # 10-second sliding window

        # Raw signal buffers
        self._green_buf: deque = deque(maxlen=self._window_size)
        self._timestamps: deque = deque(maxlen=self._window_size)

        # Public estimated values
        self.heart_rate: float = 0.0
        self.stress_level: float = 0.0
        self.temperature: float = 98.6
        self.hrv: float = 0.0

        # Filtered pulse signal for waveform display
        self._pulse_signal: deque = deque(maxlen=150)

        self._ready: bool = False
        self._start_time: float = time.time()

        # Temperature simulation state
        self._temp_phase: float = 0.0
        self._temp_base: float = 98.4

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, face_roi_frame, timestamp: float = None):
        """Process a new face ROI frame and refresh estimates.

        Args:
            face_roi_frame: BGR numpy array cropped to the face / forehead.
            timestamp: Frame capture time (defaults to ``time.time()``).
        """
        if face_roi_frame is None or face_roi_frame.size == 0:
            return

        if timestamp is None:
            timestamp = time.time()

        # Extract mean green-channel value (most sensitive to pulse)
        mean_bgr = face_roi_frame.reshape(-1, 3).mean(axis=0)
        self._green_buf.append(float(mean_bgr[1]))
        self._timestamps.append(timestamp)

        min_frames = self.sampling_rate * 5  # need ≥ 5 s for a stable estimate
        if len(self._green_buf) >= min_frames:
            self._ready = True
            self._estimate_heart_rate()
            self._estimate_stress()
            self._estimate_temperature(timestamp)

    def get_pulse_waveform(self) -> np.ndarray:
        """Return the recent filtered pulse waveform as a 1-D numpy array."""
        return np.array(list(self._pulse_signal))

    def is_ready(self) -> bool:
        """Return True once there is enough data for reliable estimates."""
        return self._ready

    def reset(self):
        """Clear all buffers and reset estimates to default values."""
        self._green_buf.clear()
        self._timestamps.clear()
        self._pulse_signal.clear()
        self.heart_rate = 0.0
        self.stress_level = 0.0
        self.temperature = 98.6
        self.hrv = 0.0
        self._ready = False
        self._start_time = time.time()
        self._temp_phase = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _estimate_heart_rate(self):
        """Estimate heart rate from the green-channel signal using FFT."""
        raw = np.array(list(self._green_buf))
        if len(raw) < self.sampling_rate * 3:
            return

        # Detrend → Hamming window → bandpass filter
        detrended = sp_signal.detrend(raw)
        windowed = detrended * np.hamming(len(detrended))

        nyquist = self.sampling_rate / 2.0
        low = self._HR_MIN_HZ / nyquist
        high = min(self._HR_MAX_HZ / nyquist, 0.99)

        try:
            b, a = sp_signal.butter(4, [low, high], btype="band")
            filtered = sp_signal.filtfilt(b, a, windowed)
        except Exception:
            filtered = windowed

        # Store the most recent 150 samples for the waveform display
        self._pulse_signal = deque(list(filtered[-150:]), maxlen=150)

        # FFT – find the dominant frequency in the valid BPM band
        freqs = np.fft.rfftfreq(len(filtered), d=1.0 / self.sampling_rate)
        fft_mag = np.abs(np.fft.rfft(filtered))

        valid = (freqs >= self._HR_MIN_HZ) & (freqs <= self._HR_MAX_HZ)
        if not valid.any():
            return

        peak_freq = freqs[valid][np.argmax(fft_mag[valid])]
        estimated_hr = peak_freq * 60.0

        # Exponential smoothing
        if self.heart_rate > 0:
            self.heart_rate = 0.7 * self.heart_rate + 0.3 * estimated_hr
        else:
            self.heart_rate = estimated_hr

    def _estimate_stress(self):
        """Estimate stress level (0–100 %) from heart-rate variability (HRV)."""
        raw = np.array(list(self._green_buf))
        if len(raw) < self.sampling_rate * 5:
            return

        detrended = sp_signal.detrend(raw)
        nyquist = self.sampling_rate / 2.0
        low = self._HR_MIN_HZ / nyquist
        high = min(self._HR_MAX_HZ / nyquist, 0.99)

        try:
            b, a = sp_signal.butter(4, [low, high], btype="band")
            filtered = sp_signal.filtfilt(b, a, detrended)
        except Exception:
            filtered = detrended

        # Detect heartbeat peaks
        min_dist = int(self.sampling_rate * 0.4)  # min 0.4 s between beats
        peaks, _ = sp_signal.find_peaks(
            filtered,
            distance=min_dist,
            height=filtered.std() * 0.5,
        )

        if len(peaks) >= 3:
            rr_ms = np.diff(peaks) / self.sampling_rate * 1000.0  # ms
            if len(rr_ms) >= 2:
                rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
                self.hrv = rmssd
                # RMSSD > 50 ms → relaxed; < 20 ms → high stress
                stress = max(0.0, min(100.0, 100.0 * (1.0 - rmssd / 50.0)))
                self.stress_level = 0.8 * self.stress_level + 0.2 * stress

    def _estimate_temperature(self, timestamp: float):
        """Simulate body temperature with a slow physiological drift.

        A real deployment would use a calibrated thermal-imaging camera.
        """
        self._temp_phase += 0.001
        variation = np.sin(self._temp_phase) * 0.3

        # Slight positive correlation with elevated heart rate (fever model)
        hr_factor = max(0.0, (self.heart_rate - 80.0) * 0.01) if self.heart_rate > 80 else 0.0

        noise = float(np.random.normal(0, 0.05))
        target = self._temp_base + variation + hr_factor + noise

        self.temperature = 0.95 * self.temperature + 0.05 * target
