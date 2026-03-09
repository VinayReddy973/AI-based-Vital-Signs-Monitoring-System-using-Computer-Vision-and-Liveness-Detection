import streamlit as st
import cv2
import numpy as np
from collections import deque
import time

st.title("AI Vital Signs Monitoring System")

run = st.checkbox("Start Camera")

frame_placeholder = st.empty()

cap = None

signal_buffer = deque(maxlen=300)
bpm_history = deque(maxlen=10)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def estimate_bpm(signal, fps=30):

    if len(signal) < fps*8:
        return None

    signal = np.array(signal)
    signal = signal - np.mean(signal)

    freqs = np.fft.rfftfreq(len(signal), d=1/fps)
    fft = np.abs(np.fft.rfft(signal))

    mask = (freqs > 0.7) & (freqs < 3)

    if not np.any(mask):
        return None

    peak = freqs[mask][np.argmax(fft[mask])]

    bpm = peak * 60

    if 40 < bpm < 180:
        return int(bpm)

    return None


if run:

    cap = cv2.VideoCapture(0)

    while run:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:

            face = frame[y:y+h, x:x+w]

            forehead = face[0:int(h*0.3), int(w*0.3):int(w*0.7)]

            green_mean = np.mean(forehead[:,:,1])

            signal_buffer.append(green_mean)

            bpm = estimate_bpm(signal_buffer)

            if bpm:
                bpm_history.append(bpm)

            pulse = int(np.median(bpm_history)) if bpm_history else None

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            if pulse:
                cv2.putText(frame,
                            f"Pulse: {pulse} BPM",
                            (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()
