import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from collections import deque
import time

st.title("AI Vital Signs Monitoring System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

signal_buffer = deque(maxlen=300)
time_buffer = deque(maxlen=300)


def bandpass(signal):
    signal = np.array(signal)
    signal = signal - np.mean(signal)
    return signal


def calculate_bpm(signal, fps=30):
    if len(signal) < 120:
        return None

    signal = bandpass(signal)

    freqs = np.fft.rfftfreq(len(signal), d=1 / fps)
    fft = np.abs(np.fft.rfft(signal))

    mask = (freqs > 0.7) & (freqs < 3)

    if not np.any(mask):
        return None

    peak = freqs[mask][np.argmax(fft[mask])]

    bpm = peak * 60

    if 40 < bpm < 180:
        return int(bpm)

    return None


def calculate_hrv(signal):
    if len(signal) < 120:
        return None

    diff = np.diff(signal)

    hrv = np.std(diff)

    return hrv


def stress_from_hrv(hrv):

    if hrv is None:
        return "Calculating"

    if hrv > 0.8:
        return "Low"

    if hrv > 0.4:
        return "Normal"

    return "High"


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            face = img[y:y + h, x:x + w]

            green_mean = np.mean(face[:, :, 1])

            signal_buffer.append(green_mean)
            time_buffer.append(time.time())

            bpm = calculate_bpm(signal_buffer)

            hrv = calculate_hrv(signal_buffer)

            stress = stress_from_hrv(hrv)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if bpm:
                cv2.putText(img, f"Heart Rate: {bpm} BPM",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2)

            cv2.putText(img, f"Stress Level: {stress}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

            cv2.putText(img, "Temp: Thermal sensor required",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vital",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
