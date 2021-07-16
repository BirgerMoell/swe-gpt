from transformers import GPT2Tokenizer, GPT2Model, FlaxGPT2LMHeadModel, GPT2LMHeadModel, pipeline, set_seed
 
tokenizer = GPT2Tokenizer.from_pretrained(".")
model = GPT2LMHeadModel.from_pretrained(".")


generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(42)
result = generator("En dag för länge sedan så fanns det", max_length=150, num_return_sequences=5)
print(result)