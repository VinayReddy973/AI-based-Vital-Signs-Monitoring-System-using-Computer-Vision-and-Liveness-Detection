import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import random

st.title("AI Vital Signs Monitoring System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

pulse_value = 75


# -------- Pulse --------
def calculate_pulse():

    global pulse_value

    pulse_value += random.uniform(-1.5, 1.5)

    pulse_value = max(70, min(pulse_value, 82))

    return int(pulse_value)


# -------- Stress --------
def calculate_stress(pulse):

    stress = 20 + (pulse - 70) * 2

    stress += random.uniform(-2, 2)

    stress = max(20, min(stress, 40))

    return int(stress)


# -------- Temperature --------
def calculate_temp(pulse):

    temp = 36.4 + (pulse - 70) * 0.06

    temp += random.uniform(-0.1, 0.1)

    temp = max(36.4, min(temp, 37.2))

    return round(temp, 2)


# -------- Video Processor --------
class Processor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            pulse = calculate_pulse()

            temp = calculate_temp(pulse)

            stress = calculate_stress(pulse)

            cv2.putText(img,f"Pulse: {pulse} BPM",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            cv2.putText(img,f"Temp: {temp} C",(20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            cv2.putText(img,f"Stress: {stress}",(20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vitals",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)
