import torch
import clip
from PIL import Image
import numpy as np

# Setup device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)

# Predefined health terms
HEALTH_TERMS = ["a wound", "a rash", "a bruise", "swelling", "infection"]
text_inputs = torch.cat([clip.tokenize(f"This image shows {desc}") for desc in HEALTH_TERMS]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

def check_frame_for_health_issue(frame: np.ndarray, threshold=0.25):
    image = Image.fromarray(frame).resize((224, 224))
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        logits_per_image = image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    matches = [(desc, prob) for desc, prob in zip(HEALTH_TERMS, probs) if prob > threshold]
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches
