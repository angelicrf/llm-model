from transformers import AutoTokenizer
import transformers
import torch

#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

def createNewFunc():
    print('started')
    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", use_fast=False)
    """ print('tokenizerIdValue: ', tokenizer.eos_token_id)
    pipeline = transformers.pipeline(
        "text-generation",
        model="codellama/CodeLlama-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        'def fibonacci(',
        do_sample=True,
        temperature=0.2,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=100,
    )
    for seq in sequences:
        print(f"My_Result: {seq['generated_text']}") """
    pipeline = transformers.pipeline(
    "text-generation",
    model="codellama/CodeLlama-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    )

    sequences = pipeline(
        'import socket\n\ndef ping_exponential_backoff(host: str):',
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

if __name__ == "__main__":
    createNewFunc()
