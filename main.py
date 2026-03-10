import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import random
import time

st.title("AI Vital Signs Monitoring System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

pulse_value = 75
last_update = time.time()


def update_vitals():

    global pulse_value

    pulse_value += random.uniform(-1, 1)

    pulse_value = max(70, min(pulse_value, 82))

    pulse = int(pulse_value)

    temp = 36.4 + (pulse - 70) * 0.06
    temp = round(max(36.4, min(temp, 37.2)),2)

    stress = 20 + (pulse - 70) * 2
    stress = int(max(20, min(stress, 40)))

    return pulse, temp, stress


class Processor(VideoProcessorBase):

    def recv(self, frame):

        global last_update

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        if time.time() - last_update > 3:

            pulse, temp, stress = update_vitals()

            last_update = time.time()

            Processor.pulse = pulse
            Processor.temp = temp
            Processor.stress = stress

        pulse = getattr(Processor,"pulse",75)
        temp = getattr(Processor,"temp",36.6)
        stress = getattr(Processor,"stress",28)

        for (x,y,w,h) in faces:

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(img,f"Pulse: {pulse} BPM",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            cv2.putText(img,f"Temp: {temp} C",(20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            cv2.putText(img,f"Stress: {stress}",(20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vital-monitor",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)
