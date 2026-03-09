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

green_signal = deque(maxlen=200)
time_buffer = deque(maxlen=200)


def calculate_bpm(signal, fps=30):

    if len(signal) < 50:
        return None

    signal = np.array(signal)
    signal = signal - np.mean(signal)

    freqs = np.fft.rfftfreq(len(signal), d=1/fps)
    fft = np.abs(np.fft.rfft(signal))

    mask = (freqs >= 0.7) & (freqs <= 4.0)

    if not np.any(mask):
        return None

    peak = freqs[mask][np.argmax(fft[mask])]

    return int(peak * 60)


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

    return round(36.5 + (bpm-70)*0.01,2)


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = img[y:y+h, x:x+w]

            green = np.mean(face[:,:,1])

            green_signal.append(green)
            time_buffer.append(time.time())

            bpm = calculate_bpm(green_signal)

            stress = estimate_stress(bpm)
            temp = estimate_temperature(bpm)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            if bpm:
                cv2.putText(img,f"Heart Rate: {bpm} BPM",(20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            if temp:
                cv2.putText(img,f"Temperature: {temp} C",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            if stress:
                cv2.putText(img,f"Stress: {stress}",(20,120),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vital",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
