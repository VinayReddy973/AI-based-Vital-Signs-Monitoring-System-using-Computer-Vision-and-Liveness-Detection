import streamlit as st
import cv2
import numpy as np
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from scipy.signal import find_peaks

st.title("AI Vital Signs Monitoring System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

signal_buffer = deque(maxlen=300)
bpm_history = deque(maxlen=10)


# ---------------- Pulse ----------------
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


# ---------------- Temperature ----------------
def estimate_temperature(bpm):

    if bpm is None:
        return None

    temp = 36.5 + (bpm - 70) * 0.01

    return round(temp,2)


# ---------------- Stress Index ----------------
def estimate_stress(bpm):

    if bpm is None:
        return None

    stress_index = (bpm - 60) / 40

    stress_index = max(0, min(stress_index,1))

    return round(stress_index,2)


# ---------------- Video Processing ----------------
class Processor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = img[y:y+h, x:x+w]

            red_signal = np.mean(face[:,:,2])

            signal_buffer.append(red_signal)

            bpm = calculate_bpm(signal_buffer)

            if bpm:
                bpm_history.append(bpm)

            pulse = int(np.median(bpm_history)) if bpm_history else None

            temp = estimate_temperature(pulse)

            stress = estimate_stress(pulse)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            if pulse:
                cv2.putText(img,f"Pulse: {pulse} BPM",(20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            if temp:
                cv2.putText(img,f"Temp: {temp} C",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            if stress is not None:
                cv2.putText(img,f"Stress Index: {stress}",
                            (20,120),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vital-monitor",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)
