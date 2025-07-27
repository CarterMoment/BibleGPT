# prepare.py

import os
import pickle
import numpy as np

# Load raw text
with open("data/clean_bible.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"Dataset length: {len(text):,} characters")

# Character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

# Create mappings from char to int and back
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Encode the entire text
data = np.array(encode(text), dtype=np.uint16)
print(f"Encoded dataset shape: {data.shape}")

# Train/val split (90/10)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data   = data[split_idx:]

# Save to binary files
train_path = os.path.join("data", "train.bin")
val_path   = os.path.join("data", "val.bin")
train_data.tofile(train_path)
val_data.tofile(val_path)

# Save vocab
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open("data/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("Done. Saved train.bin, val.bin, and meta.pkl.")
