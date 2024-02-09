from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForQuestionAnswering
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

# Sentiment Analysis Model
thisClassifier = pipeline('sentiment-analysis')
responce = thisClassifier('Hello, my dog is cute')[0]

print(responce)

modelName = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(modelName)
tokenizer = AutoTokenizer.from_pretrained(modelName)

secondClassifier = pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)
secondResponce = secondClassifier('Hello, my dog is cute')[0]

print(secondResponce)

# Convert Text to Emotion Model
thirdClassifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
newRes = thirdClassifier("I love this!")
print(newRes)

# Image to Text Model
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

readText = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png")
print(readText)

# QA Model
model_name = "deepset/roberta-base-squad2"

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'What is the purpose of a constructor in object-oriented programming?',
    'context': 'Constructors are special methods used to initialize objects of a class. They are called automatically when an object is created and are typically used to set initial values for object attributes.'
}
res = nlp(QA_input)
print('from res ',res)

model_QA = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer_QA = AutoTokenizer.from_pretrained(model_name)
res2 = nlp(QA_input)
print('from res2',res2)

""" dataset = load_dataset("cifar10")


for i, image_data in enumerate(dataset):
    img = image_data['img']
    label = image_data['label']
    
    axes[i].imshow(img)
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.show() """