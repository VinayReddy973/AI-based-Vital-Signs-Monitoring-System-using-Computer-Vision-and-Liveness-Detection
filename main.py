"""AI-based Vital Signs Monitoring System – main entry point.

Run with:
    python main.py [--camera 0] [--width 640] [--height 480]

Press 'Q' to quit, 'R' to reset all detectors.
"""

import argparse
import sys
import time

import cv2

from face_detector import FaceDetector
from health_monitor_gui import HealthMonitorGUI
from liveness_detector import LivenessDetector
from vital_signs_estimator import VitalSignsEstimator


class VitalSignsMonitorApp:
    """Main application that coordinates all monitoring modules.

    Args:
        camera_id: OpenCV camera index (default 0).
        width:     Capture width in pixels.
        height:    Capture height in pixels.
    """

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height

        self.face_detector    = FaceDetector()
        self.liveness_detector = LivenessDetector()
        self.vital_signs      = VitalSignsEstimator(sampling_rate=30)
        self.gui              = HealthMonitorGUI(frame_width=width, frame_height=height)

        self._running   = False
        self._cap       = None
        self._fps       = 0.0
        self._fps_cnt   = 0
        self._fps_timer = time.time()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self):
        """Open the camera and run the monitoring loop until the user quits."""
        if not self._open_camera():
            return

        cv2.namedWindow("Vital Signs Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vital Signs Monitor", self.gui.total_width, self.height)

        print("Vital Signs Monitor started.")
        print("  Press 'Q' to quit, 'R' to reset.\n")

        self._running = True
        try:
            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    print("Error: cannot read from camera.", file=sys.stderr)
                    break

                self._tick_fps()

                # Mirror effect for a natural selfie view
                frame = cv2.flip(frame, 1)

                face_roi = self.face_detector.detect_face(frame)

                eyes = None
                if face_roi is not None:
                    eyes = self.face_detector.detect_eyes(frame, face_roi)
                    self.liveness_detector.update(frame, face_roi, eyes)
                    forehead, _ = self.face_detector.get_forehead_roi(frame, face_roi)
                    if forehead is not None and forehead.size > 0:
                        self.vital_signs.update(forehead)

                liveness_result = (
                    self.liveness_detector.is_live,
                    self.liveness_detector.liveness_score,
                    self.liveness_detector.status_message,
                )
                vital_data = {
                    "heart_rate":   self.vital_signs.heart_rate,
                    "stress_level": self.vital_signs.stress_level,
                    "temperature":  self.vital_signs.temperature,
                }

                display = self.gui.render(
                    frame,
                    face_roi,
                    liveness_result,
                    vital_data,
                    self.vital_signs.get_pulse_waveform(),
                )

                cv2.putText(
                    display,
                    f"FPS: {self._fps:.1f}",
                    (10, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (150, 150, 150),
                    1,
                )
                cv2.imshow("Vital Signs Monitor", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or Esc
                    print("Quitting…")
                    break
                elif key in (ord("r"), ord("R")):
                    self._reset()
        finally:
            self._close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_camera(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_id)
        if not self._cap.isOpened():
            print(
                f"Error: cannot open camera {self.camera_id}.",
                file=sys.stderr,
            )
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        return True

    def _close(self):
        self._running = False
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()

    def _reset(self):
        self.liveness_detector.reset()
        self.vital_signs.reset()
        print("System reset.")

    def _tick_fps(self):
        self._fps_cnt += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps       = self._fps_cnt / elapsed
            self._fps_cnt   = 0
            self._fps_timer = now


def main():
    parser = argparse.ArgumentParser(
        description="AI-based Vital Signs Monitoring System using Computer Vision"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Capture width in pixels (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Capture height in pixels (default: 480)"
    )
    args = parser.parse_args()

    app = VitalSignsMonitorApp(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
    )
    app.run()


if __name__ == "__main__":
    main()
