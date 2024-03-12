import vec2text
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import numpy as np

def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings

encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_pretrained_corrector("gtr-base")

embeddings = get_gtr_embeddings([
       "Jack Morris is a PhD student at Cornell Tech in New York City",
       "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"
], encoder, tokenizer)

import time
t0 = time.time()
for alpha in np.arange(0.0, 1.0, 0.1):
  mixed_embedding = torch.lerp(input=embeddings[0], end=embeddings[1], weight=alpha)[None].cuda()
  text = vec2text.invert_embeddings(
      embeddings=mixed_embedding,
      corrector=corrector,
      num_steps=10,
      sequence_beam_width=4,
  )[0]
  print(f'alpha={alpha:.1f}\t', text)
print(f"Time taken: {time.time() - t0:.2f} seconds")