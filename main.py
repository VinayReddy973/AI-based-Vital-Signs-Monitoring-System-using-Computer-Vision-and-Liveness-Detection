import streamlit as st
import cv2
import numpy as np
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from scipy.signal import butter, filtfilt, find_peaks

st.title("AI Vital Signs Monitoring System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

signal_buffer = deque(maxlen=450)
bpm_history = deque(maxlen=10)


def bandpass_filter(signal, fps):

    low = 0.7
    high = 3.0
    nyq = 0.5 * fps

    b, a = butter(3, [low/nyq, high/nyq], btype='band')

    return filtfilt(b, a, signal)


def estimate_bpm(signal, fps=30):

    if len(signal) < fps * 10:
        return None

    signal = np.array(signal)

    signal = signal - np.mean(signal)

    filtered = bandpass_filter(signal, fps)

    peaks, _ = find_peaks(filtered, distance=fps/2)

    if len(peaks) < 2:
        return None

    intervals = np.diff(peaks) / fps

    bpm = 60 / np.mean(intervals)

    if 40 < bpm < 180:
        return int(bpm)

    return None


def estimate_stress(bpm):

    if bpm is None:
        return None

    if bpm < 70:
        return "Low"

    if bpm < 90:
        return "Normal"

    return "High"


def estimate_temperature(bpm):

    if bpm is None:
        return None

    return round(36.5 + (bpm - 70) * 0.01, 2)


class Processor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face = img[y:y+h, x:x+w]

            forehead = face[0:int(h*0.25), int(w*0.35):int(w*0.65)]

            green_mean = np.mean(forehead[:, :, 1])

            signal_buffer.append(green_mean)

            bpm = estimate_bpm(signal_buffer)

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

            if stress:
                cv2.putText(img,f"Stress: {stress}",(20,120),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vital-monitor",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)
