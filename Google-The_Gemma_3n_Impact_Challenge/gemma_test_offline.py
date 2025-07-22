from transformers import AutoTokenizer, AutoModelForCausalLM

# Load from local folders
tokenizer = AutoTokenizer.from_pretrained("./local_tokenizer")
model = AutoModelForCausalLM.from_pretrained("./local_model", device_map="auto")

# Generate
inputs = tokenizer("what is a transformer?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
