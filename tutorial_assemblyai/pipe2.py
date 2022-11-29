from transformers import pipeline


generator = pipeline("text-generation", model="distilgpt2")

res = generator("Using transformers is awesome!", max_length=50, num_return_sequences=3)

print(res)
