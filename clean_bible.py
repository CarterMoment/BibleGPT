import re

with open("data/bible.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Remove verse numbers like 1:1 or 12:34
cleaned = re.sub(r'\b\d+:\d+\b', '', raw_text)

# Optional: Remove double newlines and collapse weird spacing
cleaned = re.sub(r'\n+', '\n', cleaned)
cleaned = re.sub(r' +', ' ', cleaned)

with open("data/bible_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(cleaned)
