from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf",use_fast=False)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("codellama/CodeLlama-7b-hf")

# Input text
input_text = "Your input text goes here."

# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass through the model
outputs = model(**inputs)

# Extract logits (output scores) from the model's output
logits = outputs.logits

# Convert logits to probabilities using softmax
probabilities = logits.softmax(dim=1)

# Print probabilities for each class
print(probabilities)
