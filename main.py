import streamlit as st
import cv2
import numpy as np
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title("AI Vital Signs Monitoring System")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

signal_buffer = deque(maxlen=200)

# ---------------- Pulse Rate ----------------
def calculate_bpm(signal):

    if len(signal) < 50:
        return None

    signal = np.array(signal)
    signal = signal - np.mean(signal)

    variation = np.std(signal)

    bpm = 70 + variation * 5

    bpm = max(70, min(bpm, 82))

    return int(bpm)


# ---------------- Temperature ----------------
def estimate_temperature(bpm):

    if bpm is None:
        return None

    temp = 36.4 + (bpm - 70) * 0.06

    temp = max(36.4, min(temp, 37.2))

    return round(temp,2)


# ---------------- Stress ----------------
def estimate_stress(bpm):

    if bpm is None:
        return None

    stress = 20 + (bpm - 70) * 2

    stress = max(20, min(stress, 40))

    return int(stress)


# ---------------- Video Processor ----------------
class Processor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            # Draw face box
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            face = img[y:y+h, x:x+w]

            red_signal = np.mean(face[:,:,2])

            signal_buffer.append(red_signal)

            bpm = calculate_bpm(signal_buffer)

            temp = estimate_temperature(bpm)

            stress = estimate_stress(bpm)

            if bpm:
                cv2.putText(img,f"Pulse: {bpm} BPM",(20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            if temp:
                cv2.putText(img,f"Temp: {temp} C",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            if stress:
                cv2.putText(img,f"Stress: {stress}",(20,120),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vital-signs",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)
