from transformers import AutoTokenizer
import transformers
import torch
import os

os.environ["HUGGINGFACE_TOKEN"] = "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"
#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

def createNewFunc():
    print('started')
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", use_fast=False)
    print('tokenizerIdValue: ', tokenizer.eos_token_id)
    pipeline = transformers.pipeline(
        "text-generation",
        model="codellama/CodeLlama-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        'def fibonacci(',
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=100,
    )
    for seq in sequences:
        print(f"My_Result: {seq['generated_text']}")


def createModel2():
    pipeline = transformers.pipeline(
        "text-generation",
        model="codellama/CodeLlama-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto")
    myname='angy'
    input_text = 'import socket\n\ndef ping_exponential_backoff(host: str):'
    max_length = 200
    batch_size = 1  # Adjust batch size based on memory constraints
    num_return_sequences = 1 
    for i in range(0, len(input_text), batch_size):
        batch_inputs = input_text[i:i+batch_size]
        sequences = model(
        batch_inputs,
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length)
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")


if __name__ == "__main__":
    createModel2()
