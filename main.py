from transformers import AutoTokenizer, AutoModel
import torch

model_name = "xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Hello world"

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print("Embedding shape:", outputs.last_hidden_state.shape)