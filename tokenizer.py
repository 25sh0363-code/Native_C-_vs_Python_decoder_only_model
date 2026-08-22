import os
import numpy as np
import tiktoken
from tqdm import tqdm

train_file = "TinyStoriesV2-GPT4-train.txt"
valid_file = "TinyStoriesV2-GPT4-val.txt"

enc = tiktoken.get_encoding("gpt2")

def pretokenize_to_bin(filename, output_filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist. Skipping.")
        return

    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()

    # Split into individual stories and rejoin with explicit EOT separators
    stories = [s.strip() for s in data.split("<|endoftext|>") if s.strip()]
    text_with_eot = "<|endoftext|>".join(stories) + "<|endoftext|>"

    ids = enc.encode(text_with_eot, allowed_special={"<|endoftext|>"})
    ids = np.array(ids, dtype=np.uint16)

    ids.tofile(output_filename)
    print(f"Pre-tokenized {filename} to {output_filename}. Total tokens: {len(ids)}")

print("Pre-tokenizing training data...")
pretokenize_to_bin(train_file, "train.bin")
print("Pre-tokenizing validation data...")
pretokenize_to_bin(valid_file, "val.bin")
        