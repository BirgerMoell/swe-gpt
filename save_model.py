from transformers import AutoTokenizer, GPT2LMHeadModel
'''
This is a script to convert the Jax model and the tokenizer to Pytorch model
'''
model = GPT2LMHeadModel.from_pretrained(".", from_flax=True)
model.save_pretrained(".")
tokenizer = AutoTokenizer.from_pretrained(".")
tokenizer.save_pretrained(".")