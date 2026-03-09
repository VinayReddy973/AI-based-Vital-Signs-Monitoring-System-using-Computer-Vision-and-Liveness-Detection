# AI-based Vital Signs Monitoring System using Computer Vision and Liveness Detection

An AI-powered system that uses computer vision to estimate physiological signals in real time from a standard webcam. The system performs face detection, liveness detection (anti-spoofing), and estimates heart rate, stress level, and body temperature, displaying everything through a live graphical overlay with health alerts.

---

## Features

| Feature | Description |
|---|---|
| **Face Detection** | OpenCV Haar-cascade based frontal-face detection |
| **Liveness Detection** | Multi-cue anti-spoofing: eye-blink tracking, micro-movement analysis, and texture (Laplacian variance) |
| **Heart Rate (rPPG)** | Remote photoplethysmography – extracts subtle colour changes in the forehead ROI, applies a bandpass filter (0.7–3.0 Hz) and FFT to estimate BPM |
| **Stress Level** | RMSSD heart-rate variability metric derived from the rPPG signal |
| **Temperature** | Physiological simulation (thermal camera required for real measurement) |
| **Real-time GUI** | OpenCV overlay with corner-bracket face tracking, live pulse waveform, vital-signs panel |
| **Health Alerts** | Blinking on-screen banners when any value exceeds normal thresholds |

---

## Project Structure

```
├── main.py                  # Entry point – starts the monitoring application
├── face_detector.py         # Face and eye detection (Haar cascades)
├── liveness_detector.py     # Liveness / anti-spoofing detection
├── vital_signs_estimator.py # rPPG heart rate, stress level, temperature
├── health_monitor_gui.py    # Real-time GUI, waveform, alerts
├── requirements.txt         # Python dependencies
└── tests/
    └── test_modules.py      # Unit tests (pytest, no camera required)
```

---

## Requirements

- Python 3.8+
- A webcam (built-in or USB)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

Optional arguments:

```
--camera INT   Camera device index (default: 0)
--width  INT   Capture width  in pixels (default: 640)
--height INT   Capture height in pixels (default: 480)
```

**Keyboard shortcuts** (while the window is focused):

| Key | Action |
|-----|--------|
| `Q` / `Esc` | Quit |
| `R` | Reset all detectors |

---

## How It Works

### Face Detection
OpenCV's `haarcascade_frontalface_default.xml` detects the face each frame. The largest detected face is used as the region of interest (ROI).

### Liveness Detection
Three independent cues are combined into a weighted liveness score (0–1):

1. **Eye blink** (50 %) – requires at least one blink detected in a 10-second window. Photos and screens do not blink.
2. **Micro-movement** (20 %) – tracks the face centroid; live faces exhibit small natural movements.
3. **Texture** (30 %) – computes Laplacian variance of the face region; real skin has richer high-frequency texture than flat prints or screens.

A score ≥ 0.5 is classified as *live*.

### Heart Rate (rPPG)
1. Extract the forehead ROI from each frame.
2. Record the mean green-channel value (most sensitive to blood-volume changes).
3. After 5 seconds of data, detrend, apply a Hamming window, bandpass filter (0.7–3.0 Hz), and run an FFT.
4. The peak frequency in the valid BPM band is exponentially smoothed.

### Stress Level
Peak detection on the filtered rPPG signal provides beat-to-beat intervals; RMSSD (root mean square of successive differences) is computed. Higher RMSSD → lower stress.

### Health Alerts
Blinking red banners appear when:
- Heart rate < 50 BPM or > 100 BPM
- Stress level > 70 %
- Temperature < 95 °F or > 99.5 °F

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

All tests run without a camera – they use synthetic frames and signals.
