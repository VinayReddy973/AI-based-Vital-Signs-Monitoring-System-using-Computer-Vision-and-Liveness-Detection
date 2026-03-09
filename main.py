import streamlit as st
import cv2
import numpy as np
import av
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from scipy.signal import find_peaks

st.title("Phone Camera Heart Rate Monitor")

st.write("Place your finger on the phone camera with flashlight ON")

signal_buffer = deque(maxlen=300)
time_buffer = deque(maxlen=300)
bpm_history = deque(maxlen=10)


def calculate_bpm(signal, fps=30):

    if len(signal) < fps * 5:
        return None

    signal = np.array(signal)
    signal = signal - np.mean(signal)

    peaks, _ = find_peaks(signal, distance=fps/2)

    if len(peaks) < 2:
        return None

    intervals = np.diff(peaks) / fps

    bpm = 60 / np.mean(intervals)

    if 40 < bpm < 180:
        return int(bpm)

    return None


class Processor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        red_channel = np.mean(img[:,:,2])

        signal_buffer.append(red_channel)

        bpm = calculate_bpm(signal_buffer)

        if bpm:
            bpm_history.append(bpm)

        pulse = int(np.median(bpm_history)) if bpm_history else None

        if pulse:

            cv2.putText(img,
                        f"Pulse: {pulse} BPM",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="pulse",
    video_processor_factory=Processor,
    media_stream_constraints={"video": True, "audio": False},
)
