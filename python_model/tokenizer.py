import os
import numpy as np
import tiktoken
from tqdm import tqdm

train_file = "TinyStoriesV2-GPT4-train.txt"
valid_file = "TinyStoriesV2-GPT4-valid.txt"

enc = tiktoken.get_encoding("gpt2")

def pretokenize_to_bin(filename, output_filename, chunk_size=10_000_000):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist. Skipping.")
        return

    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()

    stories = [s.strip() for s in data.split("<|endoftext|>") if s.strip()]
    del data

    total_tokens = 0
    with open(output_filename, "wb") as out_f:
        buffer = []
        buffer_len = 0
        for story in stories:
            ids = enc.encode(story + "<|endoftext|>", allowed_special={"<|endoftext|>"})
            buffer.extend(ids)
            buffer_len += len(ids)
            if buffer_len >= chunk_size:
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(out_f)
                total_tokens += len(arr)
                buffer = []
                buffer_len = 0
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(out_f)
            total_tokens += len(arr)

    print(f"Pre-tokenized {filename} to {output_filename}. Total tokens: {total_tokens}")

print("Pre-tokenizing training data...")
pretokenize_to_bin(train_file, "train.bin")
print("Pre-tokenizing validation data...")
pretokenize_to_bin(valid_file, "val.bin")
        