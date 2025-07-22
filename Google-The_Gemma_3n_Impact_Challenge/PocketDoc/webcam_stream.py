import cv2
import streamlit as st

def show_webcam_stream():
    cap = cv2.VideoCapture(0)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to read from webcam.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ Updated to avoid deprecation warning
        stframe.image(frame, channels="RGB", use_container_width=True)

        if st.session_state.get("stop_webcam", False):
            break

    cap.release()
