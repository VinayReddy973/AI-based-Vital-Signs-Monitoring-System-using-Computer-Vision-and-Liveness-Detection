"""Real-time GUI overlay and health-alert system.

Renders:
* A face bounding box with corner brackets coloured green (live) or red (spoof).
* A side panel showing heart rate, stress level (with a progress bar),
  temperature, a live pulse waveform, liveness status, and key bindings.
* Blinking alert banners when any vital sign is outside its normal range.
"""

import time

import cv2
import numpy as np


class HealthMonitorGUI:
    """Draws the monitoring UI on a combined video + panel frame.

    Args:
        frame_width:  Width of the camera video region in pixels.
        frame_height: Height of the camera video region in pixels.
        panel_width:  Width of the right-side vital-signs panel in pixels.
    """

    # BGR colour palette
    _GREEN  = (0,   255, 0)
    _RED    = (0,   0,   255)
    _YELLOW = (0,   255, 255)
    _WHITE  = (255, 255, 255)
    _GREY   = (180, 180, 180)
    _DIM    = (100, 100, 100)
    _BG     = (30,  30,  30)

    # Health-alert thresholds
    HR_LOW    = 50
    HR_HIGH   = 100
    STRESS_HIGH = 70
    TEMP_LOW  = 95.0
    TEMP_HIGH = 99.5

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        panel_width: int = 280,
    ):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.panel_width  = panel_width
        self.total_width  = frame_width + panel_width

        self.alerts: list = []
        self._alert_visible: bool = True
        self._last_blink: float = time.time()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def render(
        self,
        frame,
        face_roi,
        liveness_result: tuple,
        vital_signs: dict,
        pulse_waveform,
    ) -> np.ndarray:
        """Build and return the full display frame.

        Args:
            frame:           Raw BGR video frame.
            face_roi:        (x, y, w, h) of the detected face, or None.
            liveness_result: (is_live, score, message) tuple.
            vital_signs:     Dict with keys ``heart_rate``, ``stress_level``,
                             ``temperature``.
            pulse_waveform:  1-D numpy array of the filtered pulse signal.

        Returns:
            Combined numpy array (frame_height × total_width × 3).
        """
        display = cv2.resize(frame, (self.frame_width, self.frame_height))

        if face_roi is not None:
            self._draw_face_overlay(display, face_roi, liveness_result)
        else:
            self._draw_centred_text(
                display,
                "No face detected",
                self.frame_width // 2,
                self.frame_height // 2,
                self._YELLOW,
                0.8,
            )

        panel = self._create_panel(vital_signs, pulse_waveform, liveness_result)

        combined = np.zeros((self.frame_height, self.total_width, 3), dtype=np.uint8)
        combined[:, : self.frame_width] = display
        combined[:, self.frame_width :]  = panel
        cv2.line(
            combined,
            (self.frame_width, 0),
            (self.frame_width, self.frame_height),
            (80, 80, 80),
            2,
        )

        self._update_alerts(vital_signs, liveness_result)
        self._draw_alerts(combined)
        return combined

    # ------------------------------------------------------------------
    # Private helpers – video overlay
    # ------------------------------------------------------------------

    def _draw_face_overlay(self, frame, face_roi: tuple, liveness_result: tuple):
        """Draw corner-bracket bounding box and liveness label."""
        x, y, w, h = face_roi
        is_live, score, _ = liveness_result
        colour = self._GREEN if is_live else self._RED
        t = 2
        cl = min(w, h) // 6  # corner-bracket length

        # Top-left
        cv2.line(frame, (x, y), (x + cl, y), colour, t)
        cv2.line(frame, (x, y), (x, y + cl), colour, t)
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - cl, y), colour, t)
        cv2.line(frame, (x + w, y), (x + w, y + cl), colour, t)
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + cl, y + h), colour, t)
        cv2.line(frame, (x, y + h), (x, y + h - cl), colour, t)
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - cl, y + h), colour, t)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - cl), colour, t)

        label = "LIVE" if is_live else "CHECKING"
        lc = self._GREEN if is_live else self._YELLOW
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lc, 2)
        cv2.putText(
            frame,
            f"Score: {score:.2f}",
            (x, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            self._WHITE,
            1,
        )

    # ------------------------------------------------------------------
    # Private helpers – side panel
    # ------------------------------------------------------------------

    def _create_panel(
        self, vital_signs: dict, pulse_waveform, liveness_result: tuple
    ) -> np.ndarray:
        """Compose the vital-signs side panel."""
        panel = np.full((self.frame_height, self.panel_width, 3), self._BG, dtype=np.uint8)
        pw = self.panel_width
        y = 15

        # Title
        cv2.putText(panel, "VITAL SIGNS", (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self._WHITE, 2)
        y += 30
        self._hline(panel, y)
        y += 15

        # Heart rate
        hr = vital_signs.get("heart_rate", 0.0)
        hr_col = self._RED if (hr > 0 and (hr < self.HR_LOW or hr > self.HR_HIGH)) else self._GREEN
        y = self._draw_metric(panel, "HEART RATE", f"{hr:.0f}", "BPM", hr_col, y)
        y += 10

        # Stress level + bar
        stress = vital_signs.get("stress_level", 0.0)
        stress_col = self._RED if stress > self.STRESS_HIGH else self._GREEN
        y = self._draw_metric(panel, "STRESS LEVEL", f"{stress:.0f}", "%", stress_col, y)
        bar_w = pw - 30
        cv2.rectangle(panel, (15, y), (15 + bar_w, y + 8), (60, 60, 60), -1)
        filled = max(0, int(bar_w * stress / 100.0))
        if filled > 0:
            cv2.rectangle(panel, (15, y), (15 + filled, y + 8), stress_col, -1)
        y += 18

        # Temperature
        temp = vital_signs.get("temperature", 98.6)
        temp_col = self._RED if (temp < self.TEMP_LOW or temp > self.TEMP_HIGH) else self._GREEN
        y = self._draw_metric(panel, "TEMPERATURE", f"{temp:.1f}", "\u00b0F", temp_col, y)
        y += 10

        self._hline(panel, y)
        y += 15

        # Pulse waveform
        cv2.putText(panel, "PULSE WAVEFORM", (10, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._WHITE, 1)
        y += 20
        wh = 70
        self._draw_waveform(panel, pulse_waveform, 10, y, pw - 20, wh)
        y += wh + 15

        self._hline(panel, y)
        y += 15

        # Liveness status
        is_live, score, _ = liveness_result
        cv2.putText(panel, "LIVENESS STATUS", (10, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._WHITE, 1)
        y += 25
        live_col = self._GREEN if is_live else self._RED
        cv2.putText(
            panel,
            "LIVE PERSON" if is_live else "CHECKING…",
            (10, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            live_col,
            2,
        )
        y += 30
        cv2.putText(panel, f"Score: {score:.2f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._WHITE, 1)

        # Key-binding hints at the bottom
        y = self.frame_height - 70
        self._hline(panel, y)
        y += 10
        cv2.putText(panel, "Press 'Q' to quit",  (10, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self._DIM, 1)
        cv2.putText(panel, "Press 'R' to reset", (10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.35, self._DIM, 1)

        return panel

    def _draw_metric(
        self, panel, label: str, value: str, unit: str, colour: tuple, y: int
    ) -> int:
        """Draw a labelled metric and return the new y offset."""
        cv2.putText(panel, label, (10, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self._GREY, 1)
        y += 20
        cv2.putText(panel, value, (10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
        vw = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
        cv2.putText(panel, unit, (10 + vw + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self._GREY, 1)
        return y + 35

    def _draw_waveform(self, panel, waveform, x: int, y: int, width: int, height: int):
        """Draw the pulse waveform inside a dark rectangle."""
        cv2.rectangle(panel, (x, y), (x + width, y + height), (20, 20, 20), -1)

        if waveform is None or len(waveform) < 2:
            mid = y + height // 2
            cv2.line(panel, (x, mid), (x + width, mid), (60, 60, 60), 1)
            cv2.putText(
                panel,
                "Collecting data…",
                (x + 5, y + height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                self._DIM,
                1,
            )
            return

        w_min, w_max = waveform.min(), waveform.max()
        rng = w_max - w_min
        norm = (waveform - w_min) / rng if rng > 1e-10 else np.zeros_like(waveform)

        x_step = width / len(norm)
        pts = [
            (int(x + i * x_step), int(y + height - v * (height - 4) - 2))
            for i, v in enumerate(norm)
        ]
        for i in range(len(pts) - 1):
            cv2.line(panel, pts[i], pts[i + 1], self._GREEN, 1)
        cv2.line(panel, (x, y + height // 2), (x + width, y + height // 2), (40, 40, 40), 1)

    # ------------------------------------------------------------------
    # Private helpers – alerts
    # ------------------------------------------------------------------

    def _update_alerts(self, vital_signs: dict, liveness_result: tuple):
        """Rebuild the alerts list from current vital sign values."""
        self.alerts = []

        hr = vital_signs.get("heart_rate", 0.0)
        stress = vital_signs.get("stress_level", 0.0)
        temp = vital_signs.get("temperature", 98.6)
        is_live, _, _ = liveness_result

        if hr > 0:
            if hr < self.HR_LOW:
                self.alerts.append(f"ALERT: Low Heart Rate ({hr:.0f} BPM)")
            elif hr > self.HR_HIGH:
                self.alerts.append(f"ALERT: High Heart Rate ({hr:.0f} BPM)")

        if stress > self.STRESS_HIGH:
            self.alerts.append(f"ALERT: High Stress Level ({stress:.0f}%)")

        if temp < self.TEMP_LOW:
            self.alerts.append(f"ALERT: Low Temperature ({temp:.1f}\u00b0F)")
        elif temp > self.TEMP_HIGH:
            self.alerts.append(f"ALERT: High Temperature ({temp:.1f}\u00b0F)")

        if not is_live:
            self.alerts.append("WARNING: Liveness check in progress")

        # Toggle blink visibility every 0.5 s
        now = time.time()
        if now - self._last_blink > 0.5:
            self._alert_visible = not self._alert_visible
            self._last_blink = now

    def _draw_alerts(self, frame):
        """Render blinking alert banners at the top of the combined frame."""
        if not self.alerts or not self._alert_visible:
            return

        y = 10
        for alert in self.alerts[:3]:
            (tw, th), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (5, y - 2), (tw + 15, y + th + 4), (0, 0, 180), -1)
            cv2.putText(frame, alert, (10, y + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._WHITE, 1)
            y += th + 10

    # ------------------------------------------------------------------
    # Private helpers – misc
    # ------------------------------------------------------------------

    def _hline(self, panel, y: int):
        cv2.line(panel, (10, y), (self.panel_width - 10, y), (80, 80, 80), 1)

    def _draw_centred_text(self, frame, text: str, cx: int, cy: int, colour: tuple, scale: float):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(frame, text, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2)
