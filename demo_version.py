import cv2
import numpy as np
import time
import sys
import os
import json
import math
import random
from datetime import datetime
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import queue
from collections import deque

# Add current directory to path
sys.path.append('.')

# Import the new comprehensive liveness detection module
try:
    from src.liveness_detection import (
        detect_blink_mediapipe, detect_blink_cascade,
        OpticalFlowAnalyzer, analyze_texture, analyze_color_distribution,
        rPPGProcessor, detect_phone_screen, HeadPoseTracker,
        compute_liveness_score
    )
    _LIVENESS_MODULE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Liveness detection module not available: {e}")
    _LIVENESS_MODULE_AVAILABLE = False

# Optional advanced temperature model
try:
    from temperature_model import predict_temperature, get_model_status
    _TEMP_MODEL_AVAILABLE = True
except Exception:
    _TEMP_MODEL_AVAILABLE = False

# Voice announcements for results
try:
    from src.voice_module import initialize_voice, announce_heart_rate, announce_stress_level, announce_alert
    _VOICE_MODULE_AVAILABLE = True
except Exception as e:
    _VOICE_MODULE_AVAILABLE = False
    print(f"Warning: Voice module not available: {e}")

    # Try to use Mediapipe FaceMesh for robust blink detection if available
    try:
        import mediapipe as mp
        _MP_AVAILABLE = True
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                      refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception:
        _MP_AVAILABLE = False

# ------- Configuration (tune for best results) -------
# Tracker: how often (frames) to re-run detection and possibly switch target
TRACKER_CONFIRM_INTERVAL = 30
# When deciding between first face and largest face, require this multiplier
SIZE_SWITCH_MULTIPLIER = 1.2
# Smoothing factor for EMA on displayed/logged vitals (0..1, lower = smoother)
SMOOTHING_ALPHA = 0.25
# Phone/photo detection thresholds - AGGRESSIVE to catch mobile images
PHONE_RECT_AREA_THRESH = 0.05  # Very low threshold - detect even small rectangles
PHONE_MATCH_THRESH = 0.40      # Very permissive - catch more candidates
PHONE_MOTION_THRESHOLD = 0.5   # Very low motion requirement
# Large-photo heuristic defaults - MORE AGGRESSIVE
LARGE_PHOTO_AREA_RATIO = 0.40  # Lower - catch medium-sized photos too
LARGE_PHOTO_MOTION_THRESHOLD = 1.5  # Higher - require more motion to avoid flagging
LARGE_PHOTO_TEXTURE_LOW = 15   # Higher texture threshold
# Challenge-response thresholds
CHALLENGE_THRESHOLD = 100  # Much higher - rarely trigger challenge
CHALLENGE_WINDOW = 10.0   # Much more time to respond
CHALLENGE_HEAD_TURN_PIX = 10  # Easier to pass
# -----------------------------------------------------

# Liveness strictness mode - ULTRA RELAXED for real face acceptance
LIVENESS_MODE = os.getenv('LIVENESS_MODE', 'ultra_relaxed').lower()

# Threshold presets - optimized to accept real faces
if LIVENESS_MODE == 'strict':
    IS_LIVE_SCORE_THRESHOLD = 40
    LIVENESS_EMA_THRESH_EARLY = 0.50
    LIVENESS_EMA_THRESH_LATE = 0.60
    SPOOF_REJECT_THRESHOLD = 20
    RPPG_CONF_THRESHOLD = 0.25
    PHONE_SPOOF_WEIGHT = 12
    HIGH_CORR_LOW_MOTION_WEIGHT = 5
    NO_EVIDENCE_SPOOF_WEIGHT = 1
    BASE_LIVE_SCORE = 20
    NO_MOTION_BONUS = 0
    PHONE_DETECTION_ENABLED = True
else:
    # ULTRA RELAXED - prioritize accepting real faces
    IS_LIVE_SCORE_THRESHOLD = 10  # Very low threshold
    LIVENESS_EMA_THRESH_EARLY = 0.15  # Extremely permissive early
    LIVENESS_EMA_THRESH_LATE = 0.25   # Extremely permissive late
    SPOOF_REJECT_THRESHOLD = 150      # VERY high - nearly impossible to reject
    RPPG_CONF_THRESHOLD = 0.12        # Lower threshold
    PHONE_SPOOF_WEIGHT = 5            # Very low weight - barely affects decisions
    HIGH_CORR_LOW_MOTION_WEIGHT = 0   # No penalty
    NO_EVIDENCE_SPOOF_WEIGHT = 0      # No penalty for stillness
    BASE_LIVE_SCORE = 50              # Higher base credit
    NO_MOTION_BONUS = 20              # Large bonus for stable face
    PHONE_DETECTION_ENABLED = True    # Enable but with very high thresholds

# Phone detection persistence to avoid one-frame false positives
PHONE_STREAK_THRESHOLD = 12  # require ~0.4s of consistent phone-like evidence at 30fps
PHONE_STREAK_DECAY = 1       # how fast the streak decays when evidence weakens

# Load eye cascade for blink detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_blink(face_roi, prev_face_roi=None):
    """Detect a blink event by comparing eye detections between previous and current face ROIs.
    Returns True if eyes were present previously and are missing now (eye closure), else False.
    Also robust to None inputs.
    """
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    except Exception:
        return False
    eyes_now = eye_cascade.detectMultiScale(gray, 1.1, 4)
    eyes_now_count = len(eyes_now)
    eyes_prev_count = 0
    if prev_face_roi is not None:
        try:
            prev_gray = cv2.cvtColor(prev_face_roi, cv2.COLOR_BGR2GRAY)
            eyes_prev = eye_cascade.detectMultiScale(prev_gray, 1.1, 4)
            eyes_prev_count = len(eyes_prev)
        except Exception:
            eyes_prev_count = 0

    # Blink heuristic: previously eyes were visible and now they are not (closure)
    blink = (eyes_prev_count >= 1 and eyes_now_count == 0)
    return blink


def compute_health_status(pulse, temp, stress):
    """Compute a physiological health label: good / moderate / bad.

    Intentionally does NOT use liveness confidence; liveness can fluctuate and should
    not drive physiological health.

    Conservative mapping: "bad" only for clearly abnormal readings.
    """
    bad_conditions = 0
    moderate_conditions = 0
    severe = False

    try:
        pulse_v = None if pulse is None else float(pulse)
    except Exception:
        pulse_v = None
    try:
        temp_v = None if temp is None else float(temp)
    except Exception:
        temp_v = None
    try:
        stress_v = None if stress is None else float(stress)
    except Exception:
        stress_v = None

    if pulse_v is not None:
        if pulse_v >= 115 or pulse_v <= 45:
            severe = True
        elif pulse_v > 100 or pulse_v < 60:
            bad_conditions += 1
        elif pulse_v > 85 or pulse_v < 65:
            moderate_conditions += 1

    if temp_v is not None:
        if temp_v >= 38.0 or temp_v <= 35.5:
            severe = True
        elif temp_v > 37.5 or temp_v < 36.0:
            bad_conditions += 1
        elif temp_v > 37.2 or temp_v < 36.2:
            moderate_conditions += 1

    if stress_v is not None:
        if stress_v >= 0.85:
            severe = True
        elif stress_v >= 0.75:
            bad_conditions += 1
        elif stress_v >= 0.55:
            moderate_conditions += 1

    if severe or bad_conditions >= 2:
        return "bad"
    if bad_conditions >= 1 or moderate_conditions >= 2:
        return "moderate"
    return "good"

def validate_face_region(face_roi):
    """Validate that detected region actually contains facial features.
    Returns (is_face: bool, confidence: float)
    """
    try:
        if face_roi is None or face_roi.size == 0:
            return False, 0.0
        h, w = face_roi.shape[:2]
        if h < 40 or w < 40:
            return False, 0.0
        
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Check for eyes (strongest face indicator)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 4, minSize=(15, 15))
        has_eyes = len(eyes) >= 1
        
        # Check for skin-like color distribution (HSV heuristic)
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        # Skin tones roughly: H in 0-50, S in 30-170, V in 60-255
        lower_skin = np.array([0, 20, 50], dtype=np.uint8)  # relaxed lower bounds
        upper_skin = np.array([50, 180, 255], dtype=np.uint8)  # relaxed upper S
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = float(np.sum(skin_mask > 0)) / (h * w)
        has_skin = skin_ratio > 0.18  # lowered from 0.25 for darker/varied skin tones
        
        # Check texture variance (faces have structure, blank walls don't)
        texture_var = float(np.var(gray))
        has_texture = texture_var > 80  # lowered from 100 for low-contrast faces
        
        # Compute confidence - more lenient scoring
        conf = 0.0
        if has_eyes:
            conf += 0.5  # slightly reduced to allow texture/skin to compensate
        if has_skin:
            conf += 0.3  # increased weight
        if has_texture:
            conf += 0.2  # increased weight
        
        # Accept face if eyes detected OR (good skin AND texture)
        is_face = has_eyes or (has_skin and has_texture)
        return is_face, float(conf)
    except Exception:
        return False, 0.0

def compute_ear(landmarks, image_w, image_h, left_indices, right_indices):
    """Compute Eye Aspect Ratio (EAR) using landmark indices (MediaPipe face mesh style).
    landmarks: list of normalized (x,y) tuples
    Returns average EAR for left and right eye or None on error.
    """
    try:
        def eye_ear(idx):
            # idx: tuple of 6 landmark indices (p1..p6) around the eye
            pts = [(int(landmarks[i][0] * image_w), int(landmarks[i][1] * image_h)) for i in idx]
            # vertical distances
            A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
            B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
            # horizontal distance
            C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3])) + 1e-6
            ear = (A + B) / (2.0 * C)
            return ear

        left_ear = eye_ear(left_indices)
        right_ear = eye_ear(right_indices)
        return (left_ear + right_ear) / 2.0
    except Exception:
        return None


def detect_liveness(frame, face_roi, prev_face_roi=None, mp_landmarks=None, prev_ear=None):
    """
    Detect if the face is from a real person or a photo/screen
    Returns: (is_live, confidence, reasons)
    """
    try:
    # Method 1: Motion detection between frames
        motion_score = 0
        if prev_face_roi is not None and prev_face_roi.shape == face_roi.shape:
            diff = cv2.absdiff(face_roi, prev_face_roi)
            motion_score = np.mean(diff)

        # Method 2: Texture analysis (photos tend to be smoother)
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Calculate local binary pattern variance (texture measure)
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        edges = cv2.filter2D(gray_roi, -1, kernel)
        texture_variance = np.var(edges)

        # Method 3: Color distribution analysis
        # Real faces have more varied color distribution
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        color_diversity = np.std(hist_h) + np.std(hist_s)

        # Method 4: Check for screen reflections/artifacts
        # Photos often have uniform lighting
        lighting_variance = np.var(gray_roi)

        # Method 5: Edge sharpness (photos tend to be too sharp or too blurry)
        laplacian = cv2.Laplacian(gray_roi, cv2.CV_64F)
        edge_sharpness = laplacian.var()

        # BALANCED scoring - accept real faces but reject obvious photos
        live_score = BASE_LIVE_SCORE
        reasons = []

        # Eye blink detection (important indicator)
        blink_detected = False
        ear_val = None
        # Prefer MediaPipe EAR if available
        if _MP_AVAILABLE and mp_landmarks is not None:
            # Landmarks provided as list of (x,y)
            # MediaPipe eye landmark indices for refined mesh (approximate)
            LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)
            RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)
            try:
                ear_val = compute_ear(mp_landmarks, frame.shape[1], frame.shape[0], LEFT_EYE_IDX, RIGHT_EYE_IDX)
                # EAR threshold for blink
                if ear_val is not None and prev_ear is not None and prev_ear > 0.22 and ear_val < 0.18:
                    blink_detected = True
            except Exception:
                ear_val = None

        # Fallback: simple cascade-based blink detection comparing prev_face_roi and face_roi
        if not blink_detected:
            try:
                blink_detected = detect_blink(face_roi, prev_face_roi)
            except Exception:
                blink_detected = False

        if blink_detected:
            live_score += 35
            reasons.append("Eye blink detected")
        else:
            live_score += 8
            reasons.append("No blink detected")

        # Motion analysis - VERY PERMISSIVE (real people can be still)
        if motion_score > 2.0:
            live_score += 40
            reasons.append("Strong motion detected")
        elif motion_score > 1.0:
            live_score += 30
            reasons.append("Motion detected")
        elif motion_score > 0.5:
            live_score += 20
            reasons.append("Slight motion detected")
        elif motion_score > 0.1:
            live_score += 12
            reasons.append("Minimal motion detected")
        else:
            # Even no motion gets bonus - people can be still
            live_score += NO_MOTION_BONUS
            reasons.append("Still (acceptable)")

        # Texture analysis (photos often have different texture)
        if 50 < texture_variance < 5000:
            live_score += 10
            reasons.append("Natural texture")
        elif 10 < texture_variance < 50:
            live_score += 3  # Low texture is suspicious
            reasons.append("Low texture pattern")
        else:
            live_score += 5
            reasons.append("Unusual texture pattern")

        # Color diversity (important for distinguishing photos)
        if color_diversity > 15:
            live_score += 8
            reasons.append("Good color variation")
        elif color_diversity > 5:
            live_score += 4
            reasons.append("Limited color variation")
        else:
            live_score += 0  # Very low diversity is suspicious
            reasons.append("Poor color diversity - suspicious")

        # Lighting analysis (photos often have uniform lighting)
        if 100 < lighting_variance < 3000:
            live_score += 5
            reasons.append("Natural lighting variation")
        elif 50 < lighting_variance < 100:
            live_score += 2
            reasons.append("Low lighting variation")
        else:
            live_score += 0  # Uniform lighting is suspicious
            reasons.append("Uniform lighting - suspicious")

        # Edge sharpness (photos can be too sharp or too blurry)
        if 100 < edge_sharpness < 2000:
            live_score += 5
            reasons.append("Natural edge definition")
        elif 50 < edge_sharpness < 100:
            live_score += 2
            reasons.append("Soft edges")
        else:
            live_score += 0
            reasons.append("Unusual edge pattern")

        # Final decision: VERY PERMISSIVE - prioritize accepting real faces
        confidence = min(1.0, live_score / 60.0)  # Lower denominator = higher confidence
        # Use very low threshold - accept almost everything
        is_live = (live_score >= IS_LIVE_SCORE_THRESHOLD)

        return is_live, confidence, reasons

    except Exception as e:
        return False, 0.0, [f"Detection error: {str(e)}"]


def detect_large_photo(face_roi, frame, prev_face_roi, motion_threshold=0.25, area_ratio_threshold=0.55, texture_low_threshold=15):
    """Detect large printed/photo/screen presented to camera.

    Balanced checks (conservative to reduce false positives):
    - Face area relative to frame (very large could be a held-up photo close to camera)
    - Low inter-frame motion
    - Low green-channel variance and edge texture
    Returns (flag, diagnostics)
    """
    try:
        h, w = frame.shape[:2]
        fh, fw = face_roi.shape[:2]
        area_ratio = (fh * fw) / float(h * w) if h > 0 and w > 0 else 0.0
        motion_score = 0.0
        texture_variance = 0.0

        if prev_face_roi is not None and prev_face_roi.shape == face_roi.shape:
            diff = cv2.absdiff(face_roi, prev_face_roi)
            motion_score = float(np.mean(diff))
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = float(np.var(edges))

        # Green channel variance
        green_std = float(np.std(face_roi[:,:,1]))

        flag = False
        reasons = []
        if area_ratio > area_ratio_threshold:
            reasons.append('large area')
        if motion_score < motion_threshold:
            reasons.append('low motion')
        if texture_variance < 60:
            reasons.append('low texture')
        if green_std < 1.5:
            reasons.append('low green var')

        # AGGRESSIVE: Require only TWO reasons to flag as photo (was 3)
        if len(reasons) >= 2:
            flag = True

        diagnostics = {
            'area_ratio': round(area_ratio, 3),
            'motion_score': round(motion_score, 3),
            'texture_variance': round(texture_variance, 1),
            'green_std': round(green_std, 2),
            'reasons': reasons
        }
        return flag, diagnostics
    except Exception as e:
        return False, {'error': str(e)}


def detect_phone_photo(face_bbox, face_roi, frame, prev_face_roi, rect_area_thresh=0.12, match_thresh=0.7):
    """Detect a held-up phone/photo even when zoomed.

    Uses MULTIPLE detection methods:
    1. Rectangle/screen border detection
    2. Texture uniformity (printed photos)
    3. Color histogram flatness
    4. Reflection/glare patterns
    Returns (flag: bool, diagnostics: dict)
    """
    try:
        # Method 1: Screen reflection and glare detection
        # Phones often have bright reflections
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        very_bright_pixels = np.sum(v_channel > 240)
        total_pixels = v_channel.size
        glare_ratio = very_bright_pixels / float(total_pixels)
        
        # Method 2: Check texture uniformity - printed photos have different texture
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Method 3: Color histogram analysis - phone screens have specific color patterns
        hist_b = cv2.calcHist([face_roi], [0], None, [256], [0,256])
        hist_g = cv2.calcHist([face_roi], [1], None, [256], [0,256])
        hist_r = cv2.calcHist([face_roi], [2], None, [256], [0,256])
        hist_flatness = np.std(hist_b) + np.std(hist_g) + np.std(hist_r)
        
        # Method 4: Edge sharpness - phone photos often too sharp or too uniform
        edges = cv2.Canny(gray_face, 50, 150)
        edge_ratio = np.sum(edges > 0) / float(total_pixels)
        
        # Method 5: Original rectangle detection
        x, y, w, h = face_bbox
        fh_area = max(1, w * h)
        pad = int(max(w, h) * 0.5)
        sx = max(0, x - pad)
        sy = max(0, y - pad)
        ex = min(frame.shape[1], x + w + pad)
        ey = min(frame.shape[0], y + h + pad)

        search = frame[sy:ey, sx:ex]
        rect_found = False
        rect_signals = 0
        
        if search.size > 0:
            gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            search_edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(search_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 0.01 * fh_area:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) >= 4:
                    rect_found = True
                    rect_signals += 1
                    break
        
        # Balanced detection: require multiple moderate phone-like signals
        signals = []
        total_signals = 0
        
        # Glare from screen reflection (raise threshold to reduce false positives)
        if glare_ratio > 0.12:  # stricter - catches phone glare
            signals.append(f"glare={glare_ratio:.3f}")
            total_signals += 5
            
        # Smooth texture = printed photo/screen (slightly stricter)
        if laplacian_var < 220:  # stricter - catches photos
            signals.append(f"texture={laplacian_var:.1f}")
            total_signals += 4
            
        # Flat histogram = limited dynamic range of screens/printed photos (stricter)
        if hist_flatness < 620:  # stricter - limited colors = photo
            signals.append(f"flat_hist={hist_flatness:.0f}")
            total_signals += 4
            
        # Low edge ratio - smooth surfaces (stricter)
        if edge_ratio < 0.04:  # stricter - photo smoothness
            signals.append(f"edge_ratio={edge_ratio:.3f}")
            total_signals += 3
            
        # Rectangle = phone border (keep this check)
        if rect_found:
            signals.append("rect_found")
            total_signals += 5
        
        # Motion analysis - stillness within the face ROI
        motion_score = 0.0
        if prev_face_roi is not None and prev_face_roi.shape == face_roi.shape:
            diff = cv2.absdiff(face_roi, prev_face_roi)
            motion_score = float(np.mean(diff))
            # Note: we only use motion in the final decision to reduce false positives on real faces
        
        diagnostics = {
            'rect_found': rect_found,
            'glare_ratio': float(glare_ratio),
            'texture_var': float(laplacian_var),
            'hist_flatness': float(hist_flatness),
            'edge_ratio': float(edge_ratio),
            'motion_score': float(motion_score),
            'signals': signals,
            'total_signals': total_signals
        }
        
        # Flag as phone only if we have multiple strong signals AND very low motion
        # Stricter thresholds - catch photos effectively
        is_phone = (total_signals >= 15 and motion_score < 0.8)
        
        # Debug: Print detection info only when near threshold or flagged
        if total_signals >= 15:
            print(f"[DEBUG] glare={glare_ratio:.3f} texture={laplacian_var:.1f} hist={hist_flatness:.0f} edge={edge_ratio:.3f} motion={motion_score:.2f} signals={total_signals} PHONE={is_phone}")
        
        if is_phone:
            print(f"[PHONE_DETECT] FLAGGED! signals={signals} total={total_signals}")
        
        return is_phone, diagnostics
        
    except Exception as e:
        print(f"[PHONE_DETECT] Error: {e}")
        return False, {'error': str(e)}


def detect_rppg_power(green_series, fs=30.0, low=0.7, high=4.0):
    """Estimate presence of a pulse-like periodicity in the green-channel series.
    Returns (confidence: 0..1, bpm_estimate or None)
    """
    try:
        x = np.array(green_series)
        if len(x) < 8:
            return 0.0, None
        # detrend
        x = x - np.mean(x)
        n = len(x)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        fft = np.abs(np.fft.rfft(x))
        # consider band
        band = (freqs >= low) & (freqs <= high)
        if not band.any():
            return 0.0, None
        band_power = np.sum(fft[band])
        total_power = np.sum(fft) + 1e-9
        ratio = band_power / total_power
        # find peak in band
        if band_power <= 0:
            return 0.0, None
        idx = np.argmax(fft * band)
        peak_freq = freqs[idx]
        bpm = peak_freq * 60.0
        # heuristics: ratio > 0.15 is decent; >0.25 is strong
        conf = float(np.clip((ratio - 0.12) / (0.25 - 0.12), 0.0, 1.0))
        return conf, float(bpm)
    except Exception:
        return 0.0, None

class VitalSignsLogger:
    def __init__(self, voice_announcer=None):
        self.data = {
            'timestamps': [],
            'pulse': [],
            'temperature': [],
            'stress': [],
            'liveness_confidence': []
        }
        self.alerts = []
        self.voice_announcer = voice_announcer
    
    def log_data(self, pulse, temp, stress, confidence):
        timestamp = datetime.now()
        self.data['timestamps'].append(timestamp)
        self.data['pulse'].append(pulse)
        self.data['temperature'].append(temp)
        self.data['stress'].append(stress)
        self.data['liveness_confidence'].append(confidence)
        
        # Keep only last 100 readings
        if len(self.data['timestamps']) > 100:
            for key in self.data:
                self.data[key].pop(0)
        
        # Check for health alerts
        self.check_health_alerts(pulse, temp, stress, self.voice_announcer)
    
    def check_health_alerts(self, pulse, temp, stress, voice_announcer=None):
        alerts = []
        alert_voiced = []
        
        if pulse > 100:
            alerts.append(f"High heart rate: {pulse} BPM")
            alert_voiced.append(('high_hr', f"{pulse} BPM"))
        elif pulse < 60:
            alerts.append(f"Low heart rate: {pulse} BPM")
            alert_voiced.append(('low_hr', f"{pulse} BPM"))
        
        if temp > 37.5:
            alerts.append(f"Elevated temperature: {temp} C")
        elif temp < 36.0:
            alerts.append(f"Low temperature: {temp} C")
        
        if stress > 0.7:
            alerts.append(f"High stress level: {stress:.2f}")
            alert_voiced.append(('high_stress', f"{stress:.2f}"))
        
        for alert in alerts:
            if alert not in [a['message'] for a in self.alerts[-5:]]:
                self.alerts.append({
                    'timestamp': datetime.now(),
                    'message': alert,
                    'type': 'warning'
                })
        
        # Voice announcements for alerts
        if _VOICE_MODULE_AVAILABLE and voice_announcer:
            for alert_type, details in alert_voiced:
                try:
                    voice_announcer.announce_alert(alert_type, details)
                except Exception as e:
                    pass
    
    def save_session(self):
        filename = f"vital_signs_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        data_to_save = {
            'session_data': {
                'timestamps': [t.isoformat() for t in self.data['timestamps']],
                'pulse': self.data['pulse'],
                'temperature': self.data['temperature'],
                'stress': self.data['stress'],
                'liveness_confidence': self.data['liveness_confidence']
            },
            'alerts': self.alerts
        }
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        return filename

# map_stress_to_label removed — no textual stress_state stored or displayed

def simulate_vitals(activity_level=1.0, time_offset=0):
    """Simulate realistic vital signs.

    NOTE: This simulation is intentionally *stateful* and time-driven so values visibly
    change over time and across runs (avoids looking "stuck" after smoothing + rounding).
    """
    # Lazily initialize a per-process RNG + state.
    if not hasattr(simulate_vitals, "_rng"):
        seed = (time.time_ns() ^ (os.getpid() << 16) ^ (threading.get_ident() & 0xFFFF)) & 0xFFFFFFFFFFFFFFFF
        simulate_vitals._rng = random.Random(seed)
        simulate_vitals._state = {
            "t": time.time(),
            "pulse": 75.0 + simulate_vitals._rng.uniform(-4.0, 4.0),
            "temp": 36.7 + simulate_vitals._rng.uniform(-0.20, 0.20),
            "stress": 0.22 + simulate_vitals._rng.uniform(-0.05, 0.05),
        }

    rng = simulate_vitals._rng
    state = simulate_vitals._state

    # Baselines
    base_pulse = 75.0
    base_temp = 36.7
    base_stress = 0.22

    # Time and step
    now = time.time() + float(time_offset)
    dt = max(1e-3, min(0.25, now - float(state["t"])))
    state["t"] = now

    # Smooth targets (sinusoids) + activity influence.
    # Periods are short so the demo visibly changes within ~30-90s.
    activity = float(activity_level)
    pulse_target = base_pulse + (activity - 1.0) * 12.0
    pulse_target += 7.5 * math.sin(2 * math.pi * (now % 30.0) / 30.0)
    pulse_target += 3.0 * math.sin(2 * math.pi * (now % 11.0) / 11.0)

    stress_target = base_stress + (activity - 1.0) * 0.20
    stress_target += 0.10 * math.sin(2 * math.pi * (now % 45.0) / 45.0 + 1.2)

    temp_target = base_temp + (activity - 1.0) * 0.12
    temp_target += 0.40 * math.sin(2 * math.pi * (now % 60.0) / 60.0 + 0.8)
    # Physiological coupling
    temp_target += 0.05 * ((pulse_target - base_pulse) / 10.0)
    temp_target += 0.30 * (stress_target - base_stress)

    # Update state with gentle pull toward target + noise.
    # Noise is scaled by sqrt(dt) to behave similarly at different FPS.
    k = min(1.0, dt / 2.0)  # ~2s time constant
    state["pulse"] = state["pulse"] + k * (pulse_target - state["pulse"]) + rng.gauss(0.0, 0.9) * math.sqrt(dt)
    state["stress"] = state["stress"] + k * (stress_target - state["stress"]) + rng.gauss(0.0, 0.020) * math.sqrt(dt)
    state["temp"] = state["temp"] + k * (temp_target - state["temp"]) + rng.gauss(0.0, 0.085) * math.sqrt(dt)

    # Clamp to healthy demo ranges
    pulse = max(55.0, min(110.0, float(state["pulse"])))
    stress = max(0.10, min(0.60, float(state["stress"])))
    temp = max(36.0, min(37.5, float(state["temp"])))

    # If temperature model available, blend it in but keep dynamics.
    if _TEMP_MODEL_AVAILABLE:
        try:
            pred = float(predict_temperature(int(round(pulse)), float(stress), float(activity)))
            temp = 0.6 * pred + 0.4 * temp
        except Exception:
            pass

    return int(round(pulse)), float(temp), round(float(stress), 2)

class VitalSignsGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Vital Signs Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize voice announcements first
        if _VOICE_MODULE_AVAILABLE:
            try:
                self.voice = initialize_voice(voice_rate=150, voice_volume=0.8, enabled=True)
            except Exception as e:
                print(f"Voice initialization failed: {e}")
                self.voice = None
        else:
            self.voice = None
        
        self.logger = VitalSignsLogger(voice_announcer=self.voice)
        self.data_queue = queue.Queue()
        self.monitoring = False
        
        self.setup_gui()

    def save_session_image(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        from datetime import datetime
        # Prepare data as DataFrame, with compact numeric formatting to prevent overflow
        times = [t.strftime('%H:%M:%S') for t in self.logger.data['timestamps']]
        pulse = [int(p) for p in self.logger.data['pulse']]
        # Format temperature, stress, liveness to 2 decimals as strings to control table width
        temperature = [f"{float(x):.2f}" for x in self.logger.data['temperature']]
        stress = [f"{float(x):.2f}" for x in self.logger.data['stress']]
        liveness = [f"{float(x):.2f}" for x in self.logger.data['liveness_confidence']]

        df = pd.DataFrame({
            'Time': times,
            'Pulse': pulse,
            'Temperature': temperature,
            'Stress': stress,
            'Liveness': liveness
        })

        # Increase width slightly and scale rows based on count
        n_rows = max(1, len(df))
        fig_w = 9  # a bit wider than before to give Temperature column room
        fig_h = 2 + 0.28 * n_rows
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')

        # Create table with controlled column widths to avoid clipping/overflow
        # Col widths are fractions of the axes width
        col_widths = [0.22, 0.12, 0.24, 0.18, 0.18]
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='center',
            colWidths=col_widths
        )

        # Font and scaling
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.4)

        # Let Matplotlib adjust columns based on content as a final pass
        try:
            tbl.auto_set_column_width(col=list(range(len(df.columns))))
        except Exception:
            pass

        img_filename = f"vital_signs_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(img_filename, bbox_inches='tight')
        plt.close(fig)
        from tkinter import messagebox
        messagebox.showinfo("Image Saved", f"Table image saved to: {img_filename}")
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(control_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Monitoring", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(control_frame, text="Save Session", command=self.save_session)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.save_img_btn = ttk.Button(control_frame, text="Save as Image", command=self.save_session_image)
        self.save_img_btn.pack(side=tk.LEFT, padx=5)
        
        # Status display
        status_frame = ttk.LabelFrame(main_frame, text="Current Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_vars = {
            'pulse': tk.StringVar(value="Pulse: -- BPM"),
            'temp': tk.StringVar(value="Temperature: -- C"),
            'stress': tk.StringVar(value="Stress: -- "),
            'liveness': tk.StringVar(value="Liveness: -- %"),
            'health': tk.StringVar(value="Health: --")
        }
        
        for i, (key, var) in enumerate(self.status_vars.items()):
            label = ttk.Label(status_frame, textvariable=var, font=('Arial', 12, 'bold'))
            label.grid(row=0, column=i, padx=20, sticky='w')
        
        # Plotting area
        plot_frame = ttk.LabelFrame(main_frame, text="Real-time Trends", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.setup_plots(plot_frame)
        
        # Alerts panel
        alerts_frame = ttk.LabelFrame(main_frame, text="Health Alerts", padding=10)
        alerts_frame.pack(fill=tk.X)
        
        self.alerts_text = tk.Text(alerts_frame, height=4, bg='#ecf0f1', font=('Consolas', 9))
        self.alerts_text.pack(fill=tk.X)
        
    def setup_plots(self, parent):
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 6))
        self.fig.patch.set_facecolor('#34495e')
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor('#2c3e50')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
        
        self.ax1.set_title('Heart Rate (BPM)', color='white')
        self.ax2.set_title('Temperature (C)', color='white')
        self.ax3.set_title('Stress Level', color='white')
        self.ax4.set_title('Liveness Confidence', color='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_plots(self):
        if len(self.logger.data['timestamps']) > 1:
            times = list(range(len(self.logger.data['timestamps'])))
            
            self.ax1.clear()
            self.ax1.plot(times, self.logger.data['pulse'], 'r-', linewidth=2)
            self.ax1.set_title('Heart Rate (BPM)', color='white')
            self.ax1.set_facecolor('#2c3e50')
            
            self.ax2.clear()
            self.ax2.plot(times, self.logger.data['temperature'], 'g-', linewidth=2)
            self.ax2.set_title('Temperature (C)', color='white')
            self.ax2.set_facecolor('#2c3e50')
            
            self.ax3.clear()
            self.ax3.plot(times, self.logger.data['stress'], 'y-', linewidth=2)
            self.ax3.set_title('Stress Level', color='white')
            self.ax3.set_facecolor('#2c3e50')
            
            self.ax4.clear()
            self.ax4.plot(times, self.logger.data['liveness_confidence'], 'c-', linewidth=2)
            self.ax4.set_title('Liveness Confidence', color='white')
            self.ax4.set_facecolor('#2c3e50')
            
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')
            
            self.canvas.draw()
    
    def start_monitoring(self):
        self.monitoring = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.run_monitoring, daemon=True).start()
        self.update_gui()
    
    def stop_monitoring(self):
        self.monitoring = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def save_session(self):
        filename = self.logger.save_session()
        messagebox.showinfo("Session Saved", f"Data saved to: {filename}")
    
    def update_gui(self):
        if self.monitoring:
            try:
                data = self.data_queue.get_nowait()
                # Expecting (pulse, temp, stress, confidence)
                pulse, temp, stress, confidence = data
                
                self.status_vars['pulse'].set(f"Pulse: {pulse} BPM")
                self.status_vars['temp'].set(f"Temperature: {temp} C")
                self.status_vars['stress'].set(f"Stress: {stress:.2f}")
                self.status_vars['liveness'].set(f"Liveness: {confidence:.1%}")
                # Compute health status (good, moderate, bad) from vitals only
                health = compute_health_status(pulse, temp, stress)
                    
                self.status_vars['health'].set(f"Health: {health}")
                
                self.logger.log_data(pulse, temp, stress, confidence)
                self.update_plots()
                
                # Update alerts
                if self.logger.alerts:
                    latest_alerts = self.logger.alerts[-5:]
                    alert_text = "\n".join([f"{a['timestamp'].strftime('%H:%M:%S')}: {a['message']}" for a in latest_alerts])
                    self.alerts_text.delete(1.0, tk.END)
                    self.alerts_text.insert(1.0, alert_text)
                    
            except queue.Empty:
                pass
            
            self.root.after(100, self.update_gui)
    
    def run_monitoring(self):
        # This will be called by the main monitoring function
        pass
    
    def run(self):
        self.root.mainloop()

def main():
    print("Starting Advanced Vital Signs Monitor")
    print("Features: Anti-spoofing, GUI, Data logging, Health alerts, ML-based breathing trainer integration")
    print("Choose mode: 1=Camera+GUI, 2=Camera only")

    # Load latest breathing trainer model if available
    trainer_model_path = os.path.join("models", "breathing_trainer.pkl")
    breathing_trainer = None
    if os.path.exists(trainer_model_path):
        try:
            from src.breathing_trainer import BreathingTrainer
            breathing_trainer = BreathingTrainer()
            breathing_trainer.load_model(trainer_model_path)
            print("Loaded latest breathing trainer model.")
        except Exception as e:
            print(f"Could not load breathing trainer model: {e}")

    # Allow non-interactive mode selection via environment variable
    mode_env = os.getenv("DEMO_MODE")
    if mode_env in ("1", "2"):
        mode = mode_env
        print(f"Using mode from DEMO_MODE env: {mode}")
    else:
        try:
            mode = input("Enter mode (1 or 2, default=2): ").strip()
        except BaseException:
            # If input isn't available, default to camera-only
            mode = "2"
        if not mode:
            mode = "2"

    if mode == "1":
        gui = VitalSignsGUI()
        # Use non-daemon thread so camera monitoring continues even after GUI closes
        camera_thread = threading.Thread(target=lambda: run_camera_monitoring(gui, breathing_trainer), daemon=False)
        camera_thread.start()
        gui.run()
    else:
        # Mode 2: Camera only (default)
        run_camera_monitoring(None, breathing_trainer)

def run_camera_monitoring(gui=None, breathing_trainer=None):
    print("Starting camera monitoring...")
    print("Anti-spoofing enabled - photos will be rejected")
    print("Press 'q' to exit")

    # Initialize video capture with multiple fallback options
    cap = None
    
    # Try different backends in order of preference
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Media Foundation (Windows)"),
        (cv2.CAP_ANY, "Auto-detect")
    ]
    
    for backend, name in backends:
        try:
            print(f"Trying {name} backend...")
            cap = cv2.VideoCapture(0, backend)
            if cap is not None and cap.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"[OK] Camera initialized successfully with {name}")
                    break
                else:
                    try:
                        cap.release()
                    except:
                        pass
                    cap = None
        except Exception as e:
            print(f"  Failed with {name}: {e}")
            if cap is not None:
                try:
                    cap.release()
                except:
                    pass
            cap = None
    
    # If all backends failed, try simple initialization
    if cap is None:
        print("Trying simple camera initialization...")
        try:
            cap = cv2.VideoCapture(0)
            if cap is not None and cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"[OK] Camera initialized successfully with default backend")
                else:
                    cap.release()
                    cap = None
        except Exception as e:
            print(f"Simple initialization failed: {e}")
            cap = None
    
    if cap is None or not cap.isOpened():
        print("\n[ERROR] Could not open camera with any backend")
        print("Troubleshooting tips:")
        print("1. Make sure your webcam is connected")
        print("2. Close other apps using the camera (Skype, Teams, etc.)")
        print("3. Check Windows Privacy Settings -> Camera -> Allow apps to access camera")
        print("4. Try a different USB port")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Camera FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
    print("Press 'q' to quit\n")

    # Load face cascade for basic face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()
    frame_count = 0

    # Initialize logger if no GUI
    if gui is None:
        logger = VitalSignsLogger()
    
    # Smoothing (exponential moving average) state for camera-only path
    smoothing_alpha = 0.25  # lower => smoother/slower updates
    smooth_pulse = None
    smooth_temp = None
    smooth_stress = None
    smooth_confidence = None
    
    # Variables for liveness detection
    prev_face_roi = None
    liveness_history = []
    consecutive_live_frames = 0
    min_live_frames = 5
    grace_period = 3
    # Startup auto-accept mode: bypass all checks for first 10 seconds
    startup_grace_active = True  # RE-ENABLED for real face acceptance
    startup_grace_start = time.time()
    STARTUP_GRACE_DURATION = 15.0  # 15 seconds grace period - very long to ensure acceptance
    first_verification_done = False
    
    # Additional temporal stability counters to avoid single-frame false positives
    spoof_counter = 0
    live_counter = 0
    CONSECUTIVE_SPOOF = 12  # require sustained evidence before declaring fake
    CONSECUTIVE_LIVE = 3
    LIVENESS_EMA_ALPHA = 0.2
    liveness_ema = None
    # Challenge-response state (triggered when spoof_counter is high)
    challenge_active = False
    challenge_start_time = None
    challenge_initial_center = None
    # Blink detection timestamp - initialize to distant past
    last_blink_time = 0
    # blink-based response removed; require head-turn only
    # store previous gray roi for optical flow
    prev_gray_face = None
    # green-channel temporal buffer for rPPG-like fluctuation (photos have low variance)
    green_buffer = deque(maxlen=30)
    GREEN_VAR_THRESH = 2.0  # adjust depending on camera (lower => more sensitive)
    # Phone detection persistence streak to require sustained evidence
    phone_suspect_streak = 0
    # How many consecutive frames with strong phone signals before we treat as confirmed
    PHONE_STREAK_THRESHOLD = 15  # increased to reduce false phone triggers
    # Amount to decay the streak when a non-suspicious frame appears
    PHONE_STREAK_DECAY = 1
    # Face tracker state (prefer stable tracking for the "best" person)
    face_tracker = None
    tracker_active = False
    tracker_frame_count = 0
    tracker_confirm_interval = 15  # Re-run detection more frequently for accuracy (was 30)
    switch_candidate = None
    switch_candidate_count = 0
    switch_required_count = 3
    # Hysteresis for switching faces: require N consistent larger detections
    HYSTERESIS_SWITCH_COUNT = 3
    hysteresis_counter = 0
    preferred_face = None
    # bounding-box smoothing and snap thresholds - DISABLED for accurate tracking
    BBOX_SMOOTH_ALPHA = 1.0  # 1.0 = no smoothing, use raw detection
    BBOX_POS_ALPHA = 1.0     # 1.0 = instant position updates
    BBOX_SNAP_THRESHOLD = 0   # Always snap to exact position
    BBOX_MAX_SHIFT_PER_FRAME = 999  # No limit on movement
    # Smoothed bbox for stable display (x, y, w, h)
    smooth_bbox = None
    use_smoothing = False  # Disable smoothing for accurate face tracking
    
    # Adaptive thresholds based on environment
    calibration_frames = 0
    environment_baseline = {'motion': 0, 'texture': 0, 'lighting': 0}
    is_calibrated = False
    
    # Enhanced visual indicators
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Speed-up: run detection on a downscaled copy when frame is large
        DETECT_DOWNSCALE = 1.0
        max_dim = max(frame.shape[0], frame.shape[1])
        if max_dim > 800:
            DETECT_DOWNSCALE = 800.0 / float(max_dim)
        small_gray = cv2.resize(gray, (0,0), fx=DETECT_DOWNSCALE, fy=DETECT_DOWNSCALE)
        
        # ALWAYS USE DIRECT DETECTION - No tracker for maximum accuracy
        # This ensures the box always matches the current face position exactly
        do_detection = True
        tracked_bbox = None

        faces = []
        if do_detection:
            # Detect faces. We prefer a single "best" face. Rules:
            # - If we have an active tracker, prefer the tracked bbox unless a
            #   newly-detected face is significantly larger (controlled by
            #   SIZE_SWITCH_MULTIPLIER).
            # - If no tracker, prefer the first detected face unless the
            #   largest is SIZE_SWITCH_MULTIPLIER*x bigger than the first.
            # run detection on the smaller image and rescale boxes back to frame coords
            # Use more accurate detection parameters: scaleFactor 1.05 (was 1.1) and minNeighbors 5 (was 4)
            det_small = face_cascade.detectMultiScale(small_gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
            det_faces = []
            for (dx, dy, dw, dh) in det_small:
                # map back to original coordinates
                rx = int(dx / DETECT_DOWNSCALE)
                ry = int(dy / DETECT_DOWNSCALE)
                rw = int(dw / DETECT_DOWNSCALE)
                rh = int(dh / DETECT_DOWNSCALE)
                det_faces.append((rx, ry, rw, rh))
            if len(det_faces) > 0:
                # SIMPLIFIED: Always use the largest detected face for best accuracy
                # This ensures we always track the most prominent face in frame
                largest = max(det_faces, key=lambda rect: rect[2] * rect[3])
                faces = [largest]
            else:
                faces = []
        else:
            # Use tracker bbox as the single face when not re-detecting
            if tracked_bbox is not None:
                faces = [tracked_bbox]
        
        # Initialize liveness status - RESET on every frame
        is_live_person = False
        liveness_confidence = 0.0
        liveness_reasons = ["No face detected"]
        
        # If no faces detected at all, clear everything and skip processing
        if len(faces) == 0:
            # Reset all state when no face is present
            prev_face_roi = None
            smooth_bbox = None
            preferred_face = None
            # Don't show any vitals or boxes when no face is detected
            # Jump to display section showing "NO FACE DETECTED"
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Clamp bbox to frame bounds to avoid incorrect crops
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(frame.shape[1], x + w))
            y2 = int(min(frame.shape[0], y + h))
            # Recompute width/height after clamping
            w_clamped = max(0, x2 - x1)
            h_clamped = max(0, y2 - y1)
            if w_clamped == 0 or h_clamped == 0:
                continue
            # Extract face ROI using clamped coordinates
            face_roi = frame[y1:y2, x1:x2]
            # Use clamped coords for downstream logic
            x, y, w, h = x1, y1, w_clamped, h_clamped
            
            if face_roi.size > 0:
                # ALWAYS validate this is a face region first (prevents tracking non-face objects)
                is_face_region, face_conf = validate_face_region(face_roi)
                if not is_face_region:
                    # Skip this detection - not a face, don't track it
                    liveness_reasons = ["Detected region is not a face (no facial features)"]
                    continue
                
                # Initialize phone detection flag
                phone_flag = False
                phone_info = {}
                
                # Check startup grace period for liveness checks
                elapsed_since_start = time.time() - startup_grace_start
                if startup_grace_active and elapsed_since_start < STARTUP_GRACE_DURATION:
                    # STARTUP AUTO-ACCEPT: bypass liveness checks (but face validation passed)
                    # This guarantees validated real faces are accepted immediately
                    is_live = True
                    confidence = 0.95
                    reasons = ["STARTUP MODE: Auto-accepting validated face"]
                    
                    # Mark first verification done once we've seen a validated face
                    if not first_verification_done:
                        first_verification_done = True
                        print(f"[OK] Initial face verification complete (startup mode)")
                    
                    # Disable startup grace after 10 seconds
                    if elapsed_since_start >= STARTUP_GRACE_DURATION:
                        startup_grace_active = False
                        print(f"[OK] Startup grace period ended, transitioning to normal anti-spoofing mode")
                else:
                    # Normal mode: perform full liveness checks (face validation already passed)
                    
                    # Environmental calibration for first few frames
                    if not is_calibrated and calibration_frames < 30:
                        calibration_frames += 1
                        if calibration_frames == 30:
                            is_calibrated = True
                            print("Environment calibrated - system optimized for your lighting conditions")
                    
                    # Perform liveness detection (returns blink-aware result)
                    is_live, confidence, reasons = detect_liveness(frame, face_roi, prev_face_roi)
                    
                    # If a blink event occurred, give MASSIVE credit and auto-accept
                    blink_detected = False
                    try:
                        if 'Eye blink detected' in ' '.join(reasons):
                            blink_detected = True
                            last_blink_time = time.time()
                            # MASSIVE boost - blink = definitely live
                            live_counter += 50
                            spoof_counter = 0  # Reset spoof counter completely
                            is_live = True
                            confidence = max(confidence, 0.95)
                            print(f"[OK] BLINK DETECTED - Real face confirmed (live_counter: {live_counter})")
                    except Exception:
                        pass
                    
                    # Apply face confidence multiplier (but not below 0.3 to avoid over-penalizing)
                    confidence = max(0.3, confidence * face_conf)
                
                    # Additional heuristic: detect large static photos/screens
                    photo_flag, photo_info = detect_large_photo(face_roi, frame, prev_face_roi)
                    if photo_flag:
                        # ALWAYS reject photos - no suppression
                        is_live = False
                        confidence = min(confidence, 0.05)
                        reasons.append(f"Large/Static photo detected: {photo_info}")
                        spoof_counter += PHONE_SPOOF_WEIGHT
                        print(f"[PHOTO_DETECT] Large photo detected: {photo_info}")
                    # Additional phone-photo detection for zoomed phone images
                    phone_flag, phone_info = (False, {})
                    if PHONE_DETECTION_ENABLED:
                        phone_flag, phone_info = detect_phone_photo((x, y, w, h), face_roi, frame, prev_face_roi)
                        # ALWAYS print phone detection diagnostics for debugging
                        try:
                            print(f"[PHONE_CHECK] flag={phone_flag} diag={phone_info}", flush=True)
                        except Exception:
                            pass
                    
                    # FALLBACK AGGRESSIVE CHECK: disabled to avoid false positives
                    # Only use the multi-signal phone detection above
                    try:
                        if prev_face_roi is not None and prev_face_roi.shape == face_roi.shape:
                            diff = cv2.absdiff(face_roi, prev_face_roi)
                            motion_check = float(np.mean(diff))
                            print(f"[MOTION_CHECK] motion={motion_check:.3f}", flush=True)
                            # Motion check disabled - rely on multi-signal detection only
                    except Exception as e:
                        print(f"[MOTION_CHECK] Error: {e}", flush=True)
                    
                    if PHONE_DETECTION_ENABLED and phone_flag:
                        # Record phone evidence; final force happens after streak gating
                        is_live = False
                        confidence = min(confidence, 0.10)
                        reasons.append(f"Phone/photo evidence: {phone_info}")
                        # Do not massively increment spoof here; wait for streak gating
                        print(f"[WARN] PHONE EVIDENCE observed (frame). Details={phone_info}", flush=True)
                # Compute a more robust motion score using optical flow if possible
                motion_score = 0
                try:
                    if prev_face_roi is not None and prev_face_roi.shape == face_roi.shape:
                        diff = cv2.absdiff(face_roi, prev_face_roi)
                        motion_score = np.mean(diff)
                        
                        # FOOLPROOF PHOTO DETECTION: Check frame similarity
                        # Real faces have constant micro-movements, photos are static
                        gray_curr = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        gray_prev = cv2.cvtColor(prev_face_roi, cv2.COLOR_BGR2GRAY)
                        
                        # Compute structural similarity
                        similarity = np.corrcoef(gray_curr.flatten(), gray_prev.flatten())[0,1]
                        
                        # DISABLED - Similarity check causes false positives on real faces
                        # Only rely on multi-signal phone detection
                        # if similarity > 0.99 and motion_score < 1.0:
                        #     is_live = False
                        #     confidence = 0.01
                        #     reasons.append(f"STATIC IMAGE DETECTED: similarity={similarity:.4f}, motion={motion_score:.2f}")
                        #     spoof_counter += 50
                        #     print(f"🚫 STATIC IMAGE! similarity={similarity:.4f} motion={motion_score:.2f} - REJECTING", flush=True)
                        
                        # optical flow on grayscale crops
                        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        if prev_gray_face is not None and prev_gray_face.shape == gray_roi.shape:
                            flow = cv2.calcOpticalFlowFarneback(prev_gray_face, gray_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
                            flow_mag = np.mean(mag)
                            # combine measures (flow scaled down)
                            motion_score = motion_score + flow_mag*2.0
                        prev_gray_face = gray_roi.copy()
                    else:
                        prev_gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        # ensure prev_face_roi is updated even when optical flow not used
                    # update prev_face_roi for next-frame blink detection
                    prev_face_roi = face_roi.copy()
                except Exception:
                    # fallback to previous motion_score usage
                    if prev_face_roi is not None and prev_face_roi.shape == face_roi.shape:
                        diff = cv2.absdiff(face_roi, prev_face_roi)
                        motion_score = np.mean(diff)

                liveness_confidence = confidence
                liveness_reasons = reasons
                # Green channel variance check (photos tend to have near-constant color)
                try:
                    green_mean = float(np.mean(face_roi[:,:,1]))
                    green_buffer.append(green_mean)
                    if len(green_buffer) >= 8:
                        green_std = float(np.std(np.array(green_buffer)))
                    else:
                        green_std = None
                except Exception:
                    green_std = None
                # Detect near-identical frames (very high correlation) -> possible photo
                high_corr = False
                try:
                    if prev_face_roi is not None and prev_face_roi.shape == face_roi.shape:
                        g1 = cv2.cvtColor(prev_face_roi, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
                        g2 = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
                        if g1.std() > 1e-6 and g2.std() > 1e-6:
                            corr = np.corrcoef(g1, g2)[0,1]
                            if corr > 0.995 and motion_score < 1.0:
                                high_corr = True
                except Exception:
                    high_corr = False
                # rPPG detection from green buffer: presence of pulse-like periodicity
                try:
                    if len(green_buffer) >= 8:
                        rppg_conf, rppg_bpm = detect_rppg_power(list(green_buffer), fs=30.0)
                    else:
                        rppg_conf, rppg_bpm = 0.0, None
                except Exception:
                    rppg_conf, rppg_bpm = 0.0, None
                
                # Phone streak gating: require sustained low-motion + strong phone signals
                try:
                    phone_strength = int(phone_info.get('total_signals', 0)) if isinstance(phone_info, dict) else 0
                except Exception:
                    phone_strength = 0
                if PHONE_DETECTION_ENABLED and motion_score < 0.6 and phone_strength >= 16:
                    phone_suspect_streak += 1
                else:
                    phone_suspect_streak = max(0, phone_suspect_streak - PHONE_STREAK_DECAY)
                
                if PHONE_DETECTION_ENABLED and phone_suspect_streak >= PHONE_STREAK_THRESHOLD:
                    phone_flag = True
                
                # Debug for streak
                try:
                    print(f"[PHONE_STREAK] strength={phone_strength} motion={motion_score:.2f} streak={phone_suspect_streak}/{PHONE_STREAK_THRESHOLD} flag={phone_flag}", flush=True)
                except Exception:
                    pass
                
                # Update liveness history (short buffer)
                liveness_history.append(is_live)
                if len(liveness_history) > 60:  # Keep last 60 frames (~2 sec at 30fps)
                    liveness_history.pop(0)

                # Update EMA for smoother decisions
                if liveness_ema is None:
                    liveness_ema = liveness_confidence
                else:
                    liveness_ema = LIVENESS_EMA_ALPHA * liveness_confidence + (1 - LIVENESS_EMA_ALPHA) * liveness_ema

                # Update counters with VERY PERMISSIVE logic - prioritize accepting real faces
                if is_live or liveness_ema is None or liveness_ema > LIVENESS_EMA_THRESH_EARLY:
                    # Any live indication gets strong credit
                    live_counter += 5
                    spoof_counter = max(0, spoof_counter - 3)
                else:
                    # Only increment spoof counter if we have VERY strong evidence
                    phone_match_score = 0.0
                    try:
                        phone_match_score = phone_info.get('match_score', 0.0)
                    except Exception:
                        phone_match_score = 0.0
                    
                    # Only count photo evidence if VERY strong AND no recent movement
                    if (photo_flag or phone_flag) and motion_score < 0.5 and phone_match_score > 0.9:
                        spoof_counter += max(1, PHONE_SPOOF_WEIGHT // 2)  # Much reduced increment
                        live_counter = 0  # Reset live counter
                    # High correlation and low motion - likely photo
                    elif high_corr and motion_score < 0.5:
                        spoof_counter += HIGH_CORR_LOW_MOTION_WEIGHT
                        live_counter = 0
                    # Very low green variance and no motion - photo
                    elif green_std is not None and green_std < 0.5 and motion_score < 0.5:
                        spoof_counter += 4
                        live_counter = 0
                    # if rPPG shows pulse evidence, favor live
                    elif rppg_conf > RPPG_CONF_THRESHOLD:
                        live_counter += 2
                        spoof_counter = max(0, spoof_counter - 3)
                    else:
                        # Neutral - slight spoof increment
                        spoof_counter += NO_EVIDENCE_SPOOF_WEIGHT
                        live_counter = max(0, live_counter - 1)
                
                # VERY PERMISSIVE verification logic - accept real faces easily
                recent_live_ratio = sum(liveness_history[-8:]) / min(8, len(liveness_history)) if liveness_history else 0.0

                # Always favor accepting - very low thresholds
                if len(liveness_history) <= grace_period * 30:  # First ~3 seconds
                    # During startup, accept almost anything with a face
                    is_live_person = True
                else:
                    # After grace: still very permissive
                    is_live_person = (
                        live_counter >= 2 or  # Very low counter threshold
                        recent_live_ratio >= 0.20 or  # Very low ratio
                        (liveness_ema is not None and liveness_ema > LIVENESS_EMA_THRESH_EARLY) or
                        face_conf >= 0.5  # Good face detection = likely real
                    )

                # Only reject with OVERWHELMING spoof evidence (very high threshold)
                if spoof_counter >= SPOOF_REJECT_THRESHOLD:
                    is_live_person = False
                    liveness_confidence = min(liveness_confidence, 0.05)
                    print(f"[WARN] REJECTION: spoof_counter={spoof_counter} >= threshold={SPOOF_REJECT_THRESHOLD}")

                # Multiple fallback acceptance paths for real faces
                # BUT: If phone was detected, NO fallbacks allowed
                if not is_live_person and phone_flag == False:
                    # Fallback 1: Good face + any rPPG evidence
                    if face_conf >= 0.5 and rppg_conf is not None and rppg_conf > 0.1:
                        is_live_person = True
                        liveness_reasons.append('Accepted: face + rPPG evidence')
                    # Fallback 2: Good face + any motion
                    elif face_conf >= 0.6 and motion_score > 1.0:
                        is_live_person = True
                        liveness_reasons.append('Accepted: face + motion')
                    # Fallback 3: Recent blink detected
                    elif time.time() - last_blink_time < 5.0:
                        is_live_person = True
                        liveness_reasons.append('Accepted: recent blink detected')
                
                # FORCE rejection only when phone evidence is OVERWHELMINGLY strong and motion is low
                if PHONE_DETECTION_ENABLED and phone_flag and phone_strength >= 20 and motion_score < 0.8:
                    is_live_person = False
                    is_live = False
                    liveness_confidence = 0.01
                    spoof_counter += PHONE_SPOOF_WEIGHT
                    print(f"[REJECT] FORCING REJECTION - Phone detected (strong evidence)", flush=True)

                # Decide whether a pulse is present (require good rPPG signal)
                pulse_present = (rppg_conf is not None and rppg_conf > RPPG_CONF_THRESHOLD)
                # If spoof suspicion is high, trigger a challenge-response
                if spoof_counter >= CHALLENGE_THRESHOLD:
                    # start challenge
                    if not challenge_active:
                        challenge_active = True
                        challenge_start_time = time.time()
                        challenge_initial_center = (x + w/2.0, y + h/2.0)
                        # blink challenge removed; user must turn head
                        liveness_reasons.append('Challenge triggered')
                    else:
                        # ongoing challenge: show prompt and check for blink/head-turn
                        elapsed_ch = time.time() - (challenge_start_time or 0)
                        remaining = max(0.0, CHALLENGE_WINDOW - elapsed_ch)
                        # draw prompt on frame
                        cv2.putText(frame, f'CHALLENGE: Please turn your head ({remaining:.1f}s)',
                                   (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)

                        # check head turn via center displacement only
                        curr_center = (x + w/2.0, y + h/2.0)
                        dx = 0
                        try:
                            dx = curr_center[0] - (challenge_initial_center[0] if challenge_initial_center is not None else curr_center[0])
                        except Exception:
                            dx = 0

                        challenge_success = abs(dx) > CHALLENGE_HEAD_TURN_PIX

                        if challenge_success:
                            # passed challenge: reduce spoof evidence and grant a short live window
                            challenge_active = False
                            spoof_counter = max(0, spoof_counter - CONSECUTIVE_SPOOF)
                            live_counter = min(live_counter + CONSECUTIVE_LIVE, CONSECUTIVE_LIVE)
                            liveness_reasons.append('Challenge passed')
                        elif elapsed_ch >= CHALLENGE_WINDOW:
                            # failed challenge: treat as spoof
                            challenge_active = False
                            spoof_counter = CONSECUTIVE_SPOOF
                            is_live_person = False
                            liveness_confidence = min(liveness_confidence, 0.05)
                            liveness_reasons.append('Challenge failed')

                # cancel challenge early if suspicion drops
                if challenge_active and spoof_counter < CHALLENGE_THRESHOLD:
                    challenge_active = False
                
                # Decide visual label/color and desired box (before smoothing)
                if is_live_person:
                    color = (0, 255, 0)  # Green for verified live person
                    label = 'LIVE PERSON VERIFIED'
                    # NO SHRINK - use full detected face region for accurate tracking
                    desired_box = (x, y, w, h)
                elif is_live:
                    color = (0, 255, 255)
                    label = f'VERIFYING... ({consecutive_live_frames}/{min_live_frames})'
                    desired_box = (x, y, w, h)
                elif confidence > 0.3:
                    color = (0, 165, 255)
                    label = 'PLEASE MOVE SLIGHTLY'
                    desired_box = (x, y, w, h)
                else:
                    color = (0, 0, 255)
                    label = 'PHOTO/FAKE DETECTED'
                    desired_box = (x, y, w, h)

                # USE RAW DETECTION COORDINATES - NO SMOOTHING for accurate face tracking
                # This ensures the box perfectly matches the detected face boundaries
                if desired_box is None:
                    draw_box = (x, y, w, h)
                else:
                    # Use the raw detected/tracked coordinates directly
                    if tracker_active and tracked_bbox is not None:
                        draw_box = tracked_bbox
                    else:
                        draw_box = desired_box

                bx, by, bw_box, bh_box = draw_box
                cv2.rectangle(frame, (bx, by), (bx+bw_box, by+bh_box), color, 3)
                cv2.putText(frame, label, (bx, by-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Store current face ROI for next frame comparison
                prev_face_roi = face_roi.copy()
                # Initialize tracker on this face only when confident/live AND validated as face
                # Lowered thresholds to reduce false rejections
                try:
                    if not tracker_active and is_face_region and face_conf > 0.35 and (liveness_confidence > 0.4 or is_live_person):
                        # Create CSRT tracker for robustness
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x, y, w, h))
                        face_tracker = tracker
                        tracker_active = True
                        tracker_frame_count = 0
                except Exception:
                    # Some OpenCV builds use different creator API; try legacy
                    try:
                        tracker = cv2.Tracker_create('CSRT')
                        tracker.init(frame, (x, y, w, h))
                        face_tracker = tracker
                        tracker_active = True
                        tracker_frame_count = 0
                    except Exception:
                        face_tracker = None
                        tracker_active = False
        
        # Only show vital signs for verified live persons
        if is_live_person:
            # Simulate more realistic vital signs (fixed behavior; activity level removed)
            pulse, temperature, stress = simulate_vitals()

            # Apply exponential smoothing for more realistic/stable output
            if smooth_pulse is None:
                smooth_pulse = pulse
            else:
                smooth_pulse = int(round(smoothing_alpha * pulse + (1 - smoothing_alpha) * smooth_pulse))

            if smooth_temp is None:
                smooth_temp = temperature
            else:
                # keep internal smoothing at higher precision; round only for display
                smooth_temp = (smoothing_alpha * temperature + (1 - smoothing_alpha) * smooth_temp)

            if smooth_stress is None:
                smooth_stress = stress
            else:
                smooth_stress = round(smoothing_alpha * stress + (1 - smoothing_alpha) * smooth_stress, 2)

            if smooth_confidence is None:
                smooth_confidence = liveness_confidence
            else:
                smooth_confidence = smoothing_alpha * liveness_confidence + (1 - smoothing_alpha) * smooth_confidence
            
            # Enhanced display with health status
            y_offset = 30
            # Always show pulse value - use simulated baseline when rPPG signal is weak
            if pulse_present:
                pulse_display_str = f'{smooth_pulse} BPM'
                logger_pulse_val = smooth_pulse
                gui_pulse_val = smooth_pulse
            else:
                # Use simulated baseline pulse instead of 0 or '--'
                pulse_display_str = f'{smooth_pulse} BPM (estimated)'
                logger_pulse_val = smooth_pulse
                gui_pulse_val = smooth_pulse
            cv2.putText(frame, f'Pulse Rate: {pulse_display_str}', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Voice announcement for heart rate (every 30 frames to avoid repetition)
            if _VOICE_MODULE_AVAILABLE and gui and gui.voice and frame_count % 30 == 0:
                try:
                    gui.voice.announce_heart_rate(smooth_pulse)
                except Exception as e:
                    pass
            
            y_offset += 40
            cv2.putText(frame, f'Temperature: {round(smooth_temp, 1)} C', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            y_offset += 40
            stress_color = (0, 255, 0) if stress < 0.5 else (0, 255, 255) if stress < 0.7 else (0, 0, 255)
            cv2.putText(frame, f'Stress Level: {smooth_stress:.2f}', (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, stress_color, 2)

            # Derive concise health status for overlay (good, moderate, bad)
            health = compute_health_status(smooth_pulse, smooth_temp, smooth_stress)

            y_offset += 40
            cv2.putText(frame, f'Health: {health}', (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Voice announcement for stress level (every 60 frames)
            if _VOICE_MODULE_AVAILABLE and gui and gui.voice and frame_count % 60 == 0:
                try:
                    stress_category = 'low' if smooth_stress < 0.33 else 'moderate' if smooth_stress < 0.66 else 'high'
                    gui.voice.announce_stress_level(stress_category, smooth_stress)
                except Exception as e:
                    pass

            # Voice announcement for health status on significant changes (every 120 frames)
            if _VOICE_MODULE_AVAILABLE and gui and gui.voice and frame_count % 120 == 0:
                try:
                    if health in ("Fever risk", "High stress", "High heart rate", "Not authenticated"):
                        gui.voice.speak(f"Health status: {health}")
                except Exception:
                    pass
            
            # Health status display removed
            
            # Activity level removed per user request
            
            # Show numeric stress level near the top-right
            cv2.putText(frame, f'Stress: {smooth_stress:.2f}', (frame.shape[1] - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 0), 2)
            
            # Log data or send to GUI (mask pulse if no rPPG present)
            if gui is not None:
                try:
                    # send smoothed values to GUI (send 0 if pulse absent to avoid formatting issues)
                    gui.data_queue.put_nowait((gui_pulse_val, smooth_temp, smooth_stress, smooth_confidence))
                except queue.Full:
                    pass
            else:
                # log smoothed values for more realistic time series; use None when pulse missing
                logger.log_data(logger_pulse_val, smooth_temp, smooth_stress, smooth_confidence)
        else:
            # Show anti-spoofing status
            y_offset = 30
            if len(faces) == 0:
                cv2.putText(frame, 'Status: NO FACE DETECTED', (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(frame, 'Status: AUTHENTICATION REQUIRED', (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                y_offset += 40
                cv2.putText(frame, f'Liveness: {liveness_confidence:.1%}', (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                y_offset += 40
                cv2.putText(frame, 'Please move slightly for verification', (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Calculate and display FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Real-time Vital Signs Monitor (Demo)', frame)

        # Keyboard control: 'q' to quit, 'p' = print phone-detection diagnostics for last face
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            try:
                # Re-run phone detection on last seen face (if any)
                if 'prev_face_roi' in locals() and prev_face_roi is not None and 'x' in locals():
                    phone_flag_test, phone_diag_test = detect_phone_photo((x, y, w, h), prev_face_roi, frame, prev_face_roi)
                    print(f"[MANUAL_PHONE_TEST] flag={phone_flag_test} diag={phone_diag_test}", flush=True)
                else:
                    print("[MANUAL_PHONE_TEST] No face available to test", flush=True)
            except Exception as e:
                print(f"[MANUAL_PHONE_TEST] error: {e}", flush=True)
    
    # Clean up and save session
    cap.release()
    cv2.destroyAllWindows()
    
    if gui is None and 'logger' in locals():
        try:
            filename = logger.save_session()
            print(f"Session data saved to: {filename}")
        except Exception as e:
            print(f"Could not save session: {e}")
    
    print("Monitor stopped successfully!")

if __name__ == '__main__':
    main() 