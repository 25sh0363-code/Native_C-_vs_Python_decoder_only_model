"""
Train a custom BPE tokenizer on TinyStories-train.txt.
Vocab size: 8192 (frozen spec v1).
Output: tokenizer.json (custom format: pattern + mergeable_ranks) in tokenizer/ directory.

Usage:
    python tokenizer/train_tokenizer.py
"""

import json
import time
from pathlib import Path
from rustbpe import Tokenizer
import tiktoken

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
TRAIN_FILE = DATA_DIR / "TinyStories-train.txt"
OUTPUT_FILE = TOKENIZER_DIR / "tokenizer.json"

# Frozen spec v1
VOCAB_SIZE = 8192

def main():
    TOKENIZER_DIR.mkdir(exist_ok=True)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}\n"
            "Download TinyStories-train.txt to data/ directory first."
        )

    print(f"Training BPE tokenizer on {TRAIN_FILE}")
    print(f"File size: {TRAIN_FILE.stat().st_size / 1_000_000:.1f} MB")
    print(f"Target vocab size: {VOCAB_SIZE}")

    tokenizer = Tokenizer()

    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters. Training BPE (this can take a few minutes)...")

    start = time.time()
    tokenizer.train_from_iterator(
        [text],
        vocab_size=VOCAB_SIZE,
    )
    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    # rustbpe has no native save() — export pattern + mergeable ranks ourselves
    pattern = tokenizer.get_pattern()
    mergeable_ranks = tokenizer.get_mergeable_ranks()

    # mergeable_ranks is a list of (bytes, rank) pairs -> make JSON-safe
    # bytes -> store as list of ints (or hex string), rank stays as int
    serializable_ranks = [[list(token_bytes), rank] for token_bytes, rank in mergeable_ranks]

    tokenizer_data = {
        "pattern": pattern,
        "vocab_size": VOCAB_SIZE,
        "mergeable_ranks": serializable_ranks,
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f)

    print(f"Tokenizer saved to {OUTPUT_FILE}")

    # Sanity check: rebuild a tiktoken Encoding from what we just saved and round-trip
    enc = tiktoken.Encoding(
        name="my_tokenizer",
        pat_str=pattern,
        mergeable_ranks={bytes(k): v for k, v in mergeable_ranks},
        special_tokens={},
    )

    test_text = "Once upon a time"
    tokens = enc.encode(test_text)
    decoded = enc.decode(tokens)
    print(f"Test: '{test_text}' -> tokens: {tokens} -> decoded: '{decoded}'")
    assert decoded == test_text, "Round-trip failed!"
    print("✓ Round-trip test passed.")

if __name__ == "__main__":
    main()