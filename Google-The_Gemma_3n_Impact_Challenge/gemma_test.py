# gemma_test.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "google/gemma-2b-it"  # Use 2B model for local use

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("Explain how solar panels work.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
