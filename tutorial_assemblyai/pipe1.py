from transformers import pipeline


classifier = pipeline("sentiment-analysis")

res = classifier("Using transformers is awesome!")

print(res)
