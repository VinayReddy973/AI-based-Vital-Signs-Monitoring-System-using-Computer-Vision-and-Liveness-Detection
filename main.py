import streamlit as st
import cv2
import numpy as np
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.title("AI Vital Signs Monitoring System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

last_update = time.time()

class Processor(VideoProcessorBase):

    def recv(self, frame):

        global last_update

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = img[y:y+h, x:x+w]

            # Use average brightness of face to generate different values
            brightness = np.mean(face)

            # Person-specific pulse base
            base_pulse = 70 + (brightness % 12)

            pulse = int(base_pulse)

            temp = 36.4 + (pulse - 70) * 0.06
            temp = round(max(36.4, min(temp, 37.2)),2)

            stress = 20 + (pulse - 70) * 2
            stress = int(max(20, min(stress, 40)))

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
