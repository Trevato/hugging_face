# Custom example to test the transformers library on my own

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

model = TFAutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

outputs = model.generate(**inputs)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
