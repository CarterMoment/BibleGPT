# 📜 BibleGPT

**BibleGPT** is a character-level transformer trained from scratch on the full text of the King James Bible. Inspired by Andrej Karpathy’s “GPT from scratch” tutorial, this project builds a miniature language model capable of generating biblical-sounding text — from divine proclamations to righteous hallucinations.

> _“And it came to pass, the model did utter forth tokens most holy and strange.”_

---

## ✨ Features

- Built entirely from scratch using PyTorch
- Trains on cleaned King James Bible text (`bible.txt`)
- Implements a Transformer model (self-attention, positional encoding, etc.)
- Generates new "verses" in the style of biblical scripture
- Easily extensible or repurposable for other stylized datasets

---

## 📁 Project Structure

```
BibleGPT/
├── bible.txt           # Cleaned training data (KJV)
├── prepare.py          # Tokenize and binarize data
├── train.py            # Main training loop
├── generate.py         # Text generation script
├── transformer.py      # Model architecture (Transformer)
├── config.py           # Model & training hyperparameters
├── utils.py            # (Optional) helper functions
├── train.bin / val.bin # Binary encoded datasets
├── meta.pkl            # Vocabulary + mappings
├── README.md           # This file
```

---

## 🧠 How It Works

1. `prepare.py` tokenizes the Bible at the character level and creates training/validation splits.
2. `transformer.py` defines the architecture: multi-head attention, feedforward layers, and positional encoding.
3. `train.py` performs gradient descent on batches to minimize cross-entropy loss.
4. `generate.py` uses the trained model to produce new text, character-by-character.

---

## 🚀 Getting Started

### 🧱 Install Dependencies

```bash
pip install torch numpy tqdm
```

(Optional: use a virtual environment)

### 🛠️ Prepare Data

```bash
python prepare.py
```

### 🧠 Train the Model

```bash
python train.py
```

To train for more steps, adjust `config.py` or pass args via CLI.

### 🔮 Generate Text

```bash
python generate.py
```

---

## 🧪 Sample Output

> _"And he went forth into the land of Moab, and said unto the house of Israel, Surely I will bring thee out of the wilderness."_  
>  
> _"Let not the hand of the Lord be turned away, for in the seventh hour there shall be a sign among the children of men."_

(May also produce hilariously incoherent gibberish at early training stages — it's part of the fun.)

---

## 🧰 Tips

- Use `caffeinate` on macOS to prevent sleep during training:
  ```bash
  caffeinate python train.py
  ```
- Training for **10k–50k+ steps** is recommended for coherent output.
- Model checkpoints and bin files can be large — don’t commit them to GitHub.

---

## 🛠️ Future Ideas

- Add verse/chapter-style formatting to generations
- Fine-tune on other stylized texts (e.g., Shakespeare, philosophical texts)
- Host a demo via FastAPI or Flask
- Embed into your personal website with a frontend UI

---

## 🙏 Credits

- Based on [Andrej Karpathy’s "Let's build GPT"](https://youtu.be/kCc8FmEb1nY)
- Dataset: [King James Bible (Project Gutenberg)](https://www.gutenberg.org/ebooks/10)

---

## ⚠️ Disclaimer

This is a personal/educational project. The output is not intended to be taken as real scripture or theological guidance. For entertainment and experimentation only.

---