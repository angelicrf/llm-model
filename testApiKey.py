
from openai import OpenAI
openai_api_key = "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"

client = OpenAI()
print(client)

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response)
