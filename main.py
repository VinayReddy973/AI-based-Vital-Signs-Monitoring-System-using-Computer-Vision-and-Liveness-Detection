import streamlit as st
import cv2
import numpy as np
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from scipy.signal import butter, filtfilt
import time

st.title("AI Vital Signs Monitoring System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

signal_buffer = deque(maxlen=450)
bpm_history = deque(maxlen=10)


def normalize_signal(signal):
    signal = np.array(signal)
    signal = signal - np.mean(signal)
    signal = signal / (np.std(signal) + 1e-6)
    return signal


def bandpass_filter(signal, fps):

    low = 0.7
    high = 3.0

    nyquist = 0.5 * fps

    b, a = butter(3, [low/nyquist, high/nyquist], btype="band")

    return filtfilt(b, a, signal)


def estimate_bpm(signal, fps=30):

    if len(signal) < fps*10:
        return None

    signal = normalize_signal(signal)

    filtered = bandpass_filter(signal, fps)

    freqs = np.fft.rfftfreq(len(filtered), d=1/fps)
    fft = np.abs(np.fft.rfft(filtered))

    mask = (freqs > 0.7) & (freqs < 3)

    peak = freqs[mask][np.argmax(fft[mask])]

    bpm = peak * 60

    if 40 < bpm < 180:
        return int(bpm)

    return None


class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = img[y:y+h, x:x+w]

            forehead = face[0:int(h*0.3), int(w*0.3):int(w*0.7)]

            green_mean = np.mean(forehead[:,:,1])

            signal_buffer.append(green_mean)

            bpm = estimate_bpm(signal_buffer)

            if bpm:
                bpm_history.append(bpm)

            if len(bpm_history) > 0:
                stable_bpm = int(np.median(bpm_history))
            else:
                stable_bpm = None

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            if stable_bpm:

                cv2.putText(img,
                            f"Heart Rate: {stable_bpm} BPM",
                            (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="vital-signs",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
