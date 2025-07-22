import cv2
import streamlit as st

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if not ret:
        st.error("❌ Could not capture frame from webcam.")
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    return frame
