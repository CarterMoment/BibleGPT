# generate.py

import torch
import pickle
from model.transformer import GPT
import model.config as config

# Load vocab metadata
with open("data/meta.pkl", "rb") as f:
    meta = pickle.load(f)
stoi = meta['stoi']
itos = meta['itos']
vocab_size = meta['vocab_size']

# Encoding and decoding helpers
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model_args = config
model_args.vocab_size = vocab_size
model = GPT(model_args)
model.load_state_dict(torch.load("model/oldmodel.pt", map_location=device))
model.eval().to(device)

# Prompt to start generation from (you can change this)
start_prompt = "That is super Holy"
context = torch.tensor([encode(start_prompt)], dtype=torch.long).to(device)

# Generate tokens
out = model.generate(context, max_new_tokens=500)
generated_text = decode(out[0].tolist())

print("\n📜 Generated Text:\n")
print(generated_text)
