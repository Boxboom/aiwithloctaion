from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)


print("Type 'exit' to stop chatting.")
chat_history = ""
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    chat_history += f"User: {user_input}\nAI:"

    
    inputs = tokenizer(chat_history, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=inputs['input_ids'].shape[1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )


    # Decode only the newly generated tokens, skipping the prompt
    prompt_length = inputs['input_ids'].shape[1]
    bot_reply = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    print("AI:", bot_reply)

    chat_history += f" {bot_reply}\n"