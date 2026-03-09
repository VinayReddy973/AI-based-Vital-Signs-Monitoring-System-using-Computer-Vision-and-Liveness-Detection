import streamlit as st
import cv2
import numpy as np
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import time

st.title("AI Vital Signs Monitoring System")
st.write("Real-time heart rate estimation using webcam (rPPG)")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

green_buffer = deque(maxlen=150)
time_buffer = deque(maxlen=150)

def estimate_bpm(signal, fps=30):
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
    bpm = peak * 60
    return int(bpm)


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            face = img[y:y+h, x:x+w]

            green_mean = np.mean(face[:,:,1])
            green_buffer.append(green_mean)
            time_buffer.append(time.time())

            bpm = estimate_bpm(green_buffer)

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            if bpm:
                cv2.putText(img,f"Heart Rate: {bpm} BPM",
                            (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vital-signs",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
