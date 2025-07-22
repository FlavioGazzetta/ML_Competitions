import streamlit as st
from clip_checker import check_frame_for_health_issue
import cv2

st.set_page_config(page_title="Pocket Doc", layout="centered", initial_sidebar_state="collapsed")

st.title("Pocket Doc: AI Triage Assistant")

if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    alert_box = st.empty()
    stop_button = st.button("Stop Webcam")

    frame_count = 0
    process_every = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read from webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA)

        matches = []
        if frame_count % process_every == 0:
            matches = check_frame_for_health_issue(small_frame)
        frame_count += 1

        stframe.image(frame_rgb, channels="RGB", use_container_width=True, clamp=True)

        if matches:
            alert_box.warning("⚠️ Possible issues detected:")
            issues = "\n".join([f"- **{desc}** ({prob*100:.1f}%)" for desc, prob in matches])
            alert_box.markdown(issues)
        else:
            alert_box.info("✅ No visible health issues detected.")

        if stop_button:
            break

    cap.release()