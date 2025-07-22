import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

# Load model/tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model/local_tokenizer")
    model = AutoModelForCausalLM.from_pretrained("model/local_model", device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

st.title("🩺 Pocket Doc – Offline Medical Triage Assistant")

st.markdown("Describe your symptoms and (optionally) upload a photo.")

# Text input for symptoms
symptoms = st.text_area("What symptoms are you experiencing?", height=150)

# Image input
image = st.file_uploader("Upload an image (rash, wound, etc.)", type=["jpg", "jpeg", "png"])
image_path = None

if image:
    image_path = f"images/uploaded_{image.name}"
    with open(image_path, "wb") as f:
        f.write(image.getbuffer())
    st.image(Image.open(image), caption="Uploaded image", use_column_width=True)

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        prompt = f"""You are a medical triage assistant.
The patient says: "{symptoms}"

{f'A photo of the condition has been provided.' if image else 'No image was uploaded.'}

Please assess the severity and give a recommendation. Respond clearly and calmly, and mention whether to self-manage, consult a doctor, or seek urgent care.
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=250)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown("### 🧠 Assessment")
        st.write(response)
