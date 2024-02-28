import torch
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextGenerationPipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

modelID = "cognitivecomputations/dolphin-2.2.1-mistral-7b"
# modelID = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(modelID)
model = AutoModelForCausalLM.from_pretrained(modelID, load_in_8bit="gpt2" not in modelID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
def batched_directed_bin_search(initial_point, direction_vector, covariance_matrix,
                                check_function, sample_batch_size=128, max_iters=16,
                                precision_threshold = 2**-15):
    lower_point = initial_point
    max_scale = (2**32 - 1) / np.max(np.abs(direction_vector))
    upper_point = initial_point + max_scale * direction_vector

    for _ in range(max_iters):
        mid_point = (lower_point + upper_point) / 2
        batch = np.random.multivariate_normal(mid_point, covariance_matrix, size=sample_batch_size)
        batch_results = check_function(torch.stack(batch, dim=0))

        if np.any(batch_results):
            lower_point = mid_point
        else:
            upper_point = mid_point
        component_wise_diff = np.abs(upper_point - lower_point)
        if np.all(component_wise_diff < precision_threshold):
            break
    return (lower_point + upper_point) / 2

def llm_bin_classify(query):
    prompt = [
        {"role": "system", "content": "This is a binary classification task. Reply with \"yes\" or \"no\" only, nothing else."},
        {"role": "user", "content": f"{query}\n<|im_start|>assistant"}]
    model_inputs  = tokenizer.apply_chat_template(prompt, return_tensors="pt", padding=True).to('cuda')
    generated_ids = model.generate(
                model_inputs,
                max_new_tokens=1,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id
            )
    logits = generated_ids.scores[0]
    predicted_token_ids = logits.argmax(dim=-1).tolist()
    predicted_tokens = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
    print(query, predicted_tokens)

    logits = generated_ids.scores[0][0]
    yes_index = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_index = tokenizer.encode("no", add_special_tokens=False)[0]
    relevant_logits = torch.tensor([logits[yes_index], logits[no_index]]) 
    probabilities = torch.softmax(relevant_logits, dim=-1)
    classification = probabilities[0] > probabilities[1]
    confidence = probabilities[0] if classification else probabilities[1]

    return classification.item()

def check_function(embeddings):
    if not isinstance(embeddings, torch.Tensor):  
        embeddings = torch.tensor(embeddings).to('cuda')
    return llm_bin_classify("Is the following text about desk-making in any manner?", model(embeddings))

import torch
import torch.nn as nn

class CustomEmbeddingLayer(nn.Module):
    req = None
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.hidden_size 

    def forward(self, _):
        embeddings = CustomEmbeddingLayer.req
        if CustomEmbeddingLayer.req is None:
            raise ValueError("No input found")
        else:
            print("Input found!")
        assert embeddings.shape[-1] == self.embedding_dim, \
            f"Expected embedding_dim={self.embedding_dim}, got {embeddings.shape[-1]}"

        return embeddings

if __name__ == "__main__":
    with open('desk_make.txt', 'r') as file:
        samples = file.readlines()
    samples = [sample.strip() for sample in samples]
    
    input_toks = tokenizer(samples, return_tensors="pt", padding=True)["input_ids"].to('cuda')

    if "gpt2" in modelID:
        embeddings = model.transformer.wte(input_toks) + model.transformer.wpe(input_toks)
    else:
        original_embedding = model.get_input_embeddings()
        # model.model.embed_tokens = CustomEmbeddingLayer(model.config)
        query = "Is the sky blue?"
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. Please answer the following question."},
            {"role": "user", "content": f"{query}\n<|im_start|>assistant"}]
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to('cuda')
        # CustomEmbeddingLayer.req = original_embedding(inputs['input_ids'])
        outputs = model.generate(inputs)
        input_ids = inputs['input_ids']
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_length = inputs['input_ids'].shape[1]
        generated_output = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)

        print(generated_output)

    print("Done loading")