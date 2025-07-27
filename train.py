# train.py

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle
import numpy as np
from model.transformer import GPT
import model.config as config

# Load training and validation data
train_data = np.memmap('data/train.bin', dtype=np.uint16, mode='r')
val_data   = np.memmap('data/val.bin', dtype=np.uint16, mode='r')

# Load vocab
with open("data/meta.pkl", "rb") as f:
    meta = pickle.load(f)
stoi = meta['stoi']
itos = meta['itos']
vocab_size = meta['vocab_size']

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Helper: get a batch
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Initialize model
model_args = config
model_args.vocab_size = vocab_size
model = GPT(model_args).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Training loop
for iter in range(config.max_iters):
    # Evaluate loss occasionally
    if iter % config.eval_interval == 0:
        model.eval()
        losses = {}
        for split in ['train', 'val']:
            with torch.no_grad():
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[split] = loss.item()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        model.train()

    # Get batch, forward, backward
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save final model
torch.save(model.state_dict(), "model/model.pt")
print("Training complete. Model saved to model/model.pt")
