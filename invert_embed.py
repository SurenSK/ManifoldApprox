from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

text = "Hello World!"
input_ids = tokenizer.encode(text, return_tensors="pt")

with torch.no_grad():
    token_embeddings = model.transformer.wte(input_ids)
    position_embeddings = model.transformer.wpe(torch.arange(input_ids.shape[1], device=input_ids.device))
    embeddings = token_embeddings + position_embeddings

print("Embeddings shape:", embeddings.shape)

# Infer position embeddings based on the sequence length
seq_len = embeddings.shape[1]
inferred_position_embeddings = model.transformer.wpe(torch.arange(seq_len, device=embeddings.device))

# Subtract inferred position embeddings from the total embeddings
token_embeddings = embeddings - inferred_position_embeddings

# Add Gaussian noise to the token embeddings
noise_std_dev = 0.01  # Adjust the standard deviation as needed
noise = torch.randn_like(token_embeddings) * noise_std_dev
noisy_token_embeddings = token_embeddings + noise

distances = torch.cdist(noisy_token_embeddings, model.transformer.wte.weight)
inverted_ids = torch.argmin(distances, dim=-1)
inverted_text = tokenizer.decode(inverted_ids[0])

print("Original text:", text)
print("Inverted text (with noise):", inverted_text)