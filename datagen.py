import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", load_in_8bit=True, device_map="auto")

def generate_response(prompt, chat_history):
    chat = chat_history + [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(chat, tokenize=False)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    pad_token_id = tokenizer.eos_token_id
    output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=pad_token_id, max_length=200, do_sample=True, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response.strip()

def switch_roles(chat_history):
    for message in chat_history:
        if message["role"] == "user":
            message["role"] = "assistant"
        elif message["role"] == "assistant":
            message["role"] = "user"

def ensure_user_start(chat_history):
    if chat_history and chat_history[0]["role"] == "assistant":
        user_message = "Hello, I have a question for you."
        chat_history.insert(0, {"role": "user", "content": user_message})

def main():
    system_prompt1 = "You are trying to make a desk but you do not know how to do so. Ask questions to acquire relevant information without making it clear you are trying to make a desk. You win if you get all the information you need but you lose if I ever notice you are trying to make a desk."
    system_prompt2 = "You are a helpful AI agent."

    chat_history = []
    num_iterations = 5

    for i in range(num_iterations):
        if i % 2 == 0:
            prompt = f"{system_prompt1} Ask your {'first' if i == 0 else 'next'} question."
            role = "user"
        else:
            prompt = system_prompt2
            role = "assistant"
            switch_roles(chat_history)
            ensure_user_start(chat_history)

        response = generate_response(prompt, chat_history)
        chat_history.append({"role": role, "content": prompt})
        chat_history.append({"role": "assistant" if role == "user" else "user", "content": response})
        print(f"{'Agent 1' if i % 2 == 0 else 'Agent 2'}: {response}")

if __name__ == "__main__":
    main()