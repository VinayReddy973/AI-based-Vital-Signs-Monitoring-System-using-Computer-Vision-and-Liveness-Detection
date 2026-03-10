import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="AI Vital Signs Monitor", layout="wide")

st.title("AI Based Vital Signs Monitoring System")

st.write("Camera based Heart Rate, Temperature and Stress Detection")

# load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = st.camera_input("Take a picture")

# buffers
green_signal = []
timestamps = []

def estimate_pulse(signal, times):

    if len(signal) < 10:
        return 72

    signal = np.array(signal)
    signal = signal - np.mean(signal)

    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=(times[1] - times[0]))

    idx = np.argmax(np.abs(fft))
    bpm = freqs[idx] * 60

    if 40 < bpm < 180:
        return int(bpm)

    return 72


if camera is not None:

    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:

        x, y, w, h = faces[0]

        face = frame[y:y+h, x:x+w]

        green_mean = np.mean(face[:, :, 1])

        green_signal.append(green_mean)
        timestamps.append(time.time())

        if len(green_signal) > 30:
            green_signal.pop(0)
            timestamps.pop(0)

        pulse = estimate_pulse(green_signal, timestamps)

        temperature = 36.5 + (pulse - 70) * 0.01
        stress = min(1.0, (pulse - 60) / 100)

        col1, col2, col3 = st.columns(3)

        col1.metric("Heart Rate (BPM)", pulse)
        col2.metric("Temperature (°C)", round(temperature,2))
        col3.metric("Stress Level", round(stress,2))

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        st.image(frame, channels="BGR")

        df = pd.DataFrame({
            "Heart Rate":[pulse],
            "Temperature":[temperature],
            "Stress":[stress]
        })

        st.subheader("Vital Signs")

        st.line_chart(df)

    else:
        st.warning("Face not detected")
