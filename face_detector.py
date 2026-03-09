"""Face and eye detection using OpenCV Haar cascades."""

import cv2
import numpy as np


class FaceDetector:
    """Detects faces and facial features using OpenCV Haar cascades."""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def detect_face(self, frame):
        """Detect the largest face in the frame.

        Args:
            frame: BGR image array.

        Returns:
            (x, y, w, h) tuple of the largest detected face, or None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        if len(faces) == 0:
            return None

        # Return the largest face by area
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces[0]

    def detect_eyes(self, frame, face_roi):
        """Detect eyes within the face region of interest.

        Args:
            frame: Full BGR image.
            face_roi: (x, y, w, h) tuple of the face region.

        Returns:
            Array of eye detections (may be empty).
        """
        x, y, w, h = face_roi
        face_gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)

        # Search only in the upper half of the face where eyes are located
        upper_face = face_gray[: h // 2, :]
        eyes = self.eye_cascade.detectMultiScale(
            upper_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )
        return eyes

    def get_forehead_roi(self, frame, face_roi):
        """Extract the forehead region for rPPG measurement.

        The forehead (roughly top 10–35 % of the face, centred horizontally)
        has a good signal-to-noise ratio for remote photoplethysmography.

        Args:
            frame: Full BGR image.
            face_roi: (x, y, w, h) tuple of the face region.

        Returns:
            Tuple of (roi_frame, (rx, ry, rw, rh)).  roi_frame may be empty
            if the bounding box falls outside the image.
        """
        x, y, w, h = face_roi
        fh_x = x + w // 4
        fh_y = y + h // 10
        fh_w = w // 2
        fh_h = h // 4

        # Clamp to image bounds
        img_h, img_w = frame.shape[:2]
        fh_x = max(0, min(fh_x, img_w - 1))
        fh_y = max(0, min(fh_y, img_h - 1))
        fh_w = max(1, min(fh_w, img_w - fh_x))
        fh_h = max(1, min(fh_h, img_h - fh_y))

        return frame[fh_y : fh_y + fh_h, fh_x : fh_x + fh_w], (fh_x, fh_y, fh_w, fh_h)

    def get_cheek_roi(self, frame, face_roi):
        """Extract the left-cheek region as an alternative rPPG measurement site.

        Args:
            frame: Full BGR image.
            face_roi: (x, y, w, h) tuple of the face region.

        Returns:
            Tuple of (roi_frame, (rx, ry, rw, rh)).
        """
        x, y, w, h = face_roi
        cx = x + w // 8
        cy = y + h // 3
        cw = w // 3
        ch = h // 3

        # Clamp to image bounds
        img_h, img_w = frame.shape[:2]
        cx = max(0, min(cx, img_w - 1))
        cy = max(0, min(cy, img_h - 1))
        cw = max(1, min(cw, img_w - cx))
        ch = max(1, min(ch, img_h - cy))

        return frame[cy : cy + ch, cx : cx + cw], (cx, cy, cw, ch)
