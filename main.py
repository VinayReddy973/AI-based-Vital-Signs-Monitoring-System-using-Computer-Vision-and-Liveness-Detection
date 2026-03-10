import streamlit as st
import cv2
import numpy as np
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from scipy.signal import find_peaks

st.title("AI Vital Signs Monitoring System")

st.write("Place your finger on the phone camera with flashlight ON")

signal_buffer = deque(maxlen=300)
bpm_history = deque(maxlen=10)


# ---------- Heart Rate ----------
def calculate_bpm(signal, fps=30):

    if len(signal) < fps * 5:
        return None

    signal = np.array(signal)
    signal = signal - np.mean(signal)

    peaks, _ = find_peaks(signal, distance=fps/2)

    if len(peaks) < 2:
        return None

    intervals = np.diff(peaks) / fps

    bpm = 60 / np.mean(intervals)

    if 40 < bpm < 180:
        return int(bpm)

    return None


# ---------- Temperature ----------
def estimate_temperature(bpm):

    if bpm is None:
        return None

    # estimated body temperature
    temp = 36.5 + (bpm - 70) * 0.01

    return round(temp, 2)


# ---------- Stress ----------
def estimate_stress(bpm):

    if bpm is None:
        return None

    if bpm < 65:
        return "Low"

    elif bpm < 85:
        return "Normal"

    else:
        return "High"


# ---------- Video Processor ----------
class Processor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        red_signal = np.mean(img[:, :, 2])

        signal_buffer.append(red_signal)

        bpm = calculate_bpm(signal_buffer)

        if bpm:
            bpm_history.append(bpm)

        pulse = int(np.median(bpm_history)) if bpm_history else None

        temp = estimate_temperature(pulse)
        stress = estimate_stress(pulse)

        if pulse:
            cv2.putText(img, f"Pulse: {pulse} BPM",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,0),2)

        if temp:
            cv2.putText(img, f"Temp: {temp} C",
                        (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,0),2)

        if stress:
            cv2.putText(img, f"Stress: {stress}",
                        (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="monitor",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)
