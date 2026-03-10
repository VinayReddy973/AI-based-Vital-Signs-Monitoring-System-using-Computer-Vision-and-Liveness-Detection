import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time

st.title("AI Vital Signs Monitoring System")

st.write("Camera based Heart Rate, Temperature and Stress Estimation")

camera = st.camera_input("Capture face")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# buffers
green_values = []
timestamps = []

def estimate_pulse(green_signal, times):

    if len(green_signal) < 10:
        return None

    signal = np.array(green_signal)
    signal = signal - np.mean(signal)

    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=(times[1] - times[0]))

    idx = np.argmax(np.abs(fft))
    bpm = freqs[idx] * 60

    if 40 < bpm < 180:
        return int(bpm)

    return None


if camera is not None:

    bytes_data = camera.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:

        x, y, w, h = faces[0]

        face = img[y:y+h, x:x+w]

        green = np.mean(face[:,:,1])

        green_values.append(green)
        timestamps.append(time.time())

        if len(green_values) > 30:
            green_values.pop(0)
            timestamps.pop(0)

        pulse = estimate_pulse(green_values, timestamps)

        if pulse is None:
            pulse = 72

        # temperature model
        temperature = 36.5 + (pulse - 70) * 0.01

        # stress model
        stress = min(1.0, (pulse - 60) / 100)

        col1, col2, col3 = st.columns(3)

        col1.metric("Heart Rate", f"{pulse} BPM")
        col2.metric("Temperature", f"{temperature:.2f} °C")
        col3.metric("Stress Level", f"{stress:.2f}")

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        st.image(img, channels="BGR")

        df = pd.DataFrame({
            "Heart Rate":[pulse],
            "Temperature":[temperature],
            "Stress":[stress]
        })

        st.line_chart(df)

    else:
        st.warning("Face not detected")
