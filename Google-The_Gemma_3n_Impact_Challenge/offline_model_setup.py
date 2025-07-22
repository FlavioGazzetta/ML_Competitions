from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-2b-it"

# Save locally
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./local_tokenizer")

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.save_pretrained("./local_model")
