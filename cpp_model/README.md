# TinyStories GPT — C++ port

A C++ port of your PyTorch `model.py` / `tokenizer.py` / `test.py`, built on
**LibTorch** (the PyTorch C++ API) so you keep autograd, CUDA, and the same
optimizer/training machinery instead of reimplementing them by hand. Same
`GPTConfig`, same hyperparameters, same training loop shape, same
architecture (tied embedding/output weights, causal self-attention, GELU
MLP blocks).

```
gpt_cpp/
  CMakeLists.txt
  include/
    gpt_model.h        # GPTConfig, CausalSelfAttention, Block, GPT
    bin_dataset.h       # mmap'd uint16 token file -> random (x, y) batches
    bpe_tokenizer.h      # GPT-2 byte-level BPE tokenizer
  src/
    gpt_model.cpp
    bpe_tokenizer.cpp
    tokenize_data.cpp   # -> train.bin / val.bin        (was tokenizer.py)
    train.cpp           # -> TinyStoriesV2-GPT4-model.pt (was model.py)
    generate.cpp         # -> generation_output.txt      (was test.py)
```

## 1. Prerequisites

- CMake ≥ 3.18, a C++17 compiler
- **LibTorch** — download the matching build (CPU or CUDA) from
  https://pytorch.org/get-started/locally/ ("LibTorch" under "Your OS"), and
  unzip it somewhere, e.g. `~/libtorch`.
- Internet access the first time you configure CMake (it fetches
  `nlohmann/json`, a single-header JSON library, via `FetchContent`, used
  only to parse `encoder.json`).
- The GPT-2 BPE vocabulary files (same vocab tiktoken's `"gpt2"` encoding
  uses under the hood):
  ```
  curl -O https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json
  curl -O https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe
  ```
  Put both next to your training/inference binaries (or pass paths as CLI args).

## 2. Build

```bash
cmake -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch
cmake --build build -j
```

This produces `build/tokenize_data`, `build/train`, `build/generate`.

## 3. Run

```bash
# 1. Pre-tokenize your corpus (expects TinyStoriesV2-GPT4-train.txt /
#    TinyStoriesV2-GPT4-valid.txt in the working directory, same as tokenizer.py)
./build/tokenize_data encoder.json vocab.bpe

# 2. Train (reads train.bin / val.bin, writes TinyStoriesV2-GPT4-model.pt)
./build/train             # add --amp to enable CUDA autocast

# 3. Generate (loads TinyStoriesV2-GPT4-model.pt)
./build/generate encoder.json vocab.bpe
```

## Honest differences from the Python version

These are deliberate, documented simplifications rather than hidden bugs —
worth knowing before you rely on this for anything beyond TinyStories-style
English text:

- **BPE pretokenizer regex is ASCII-only.** The real GPT-2/tiktoken regex
  uses Unicode property escapes (`\p{L}`, `\p{N}`) that `std::regex` can't
  express. This version substitutes `[A-Za-z]` / `[0-9]`. For plain English
  text (TinyStories) the output is identical to tiktoken; it only diverges
  on non-ASCII input (accents, CJK, emoji, etc.). If you need exact
  Unicode-correct behavior, swap in a PCRE2/ICU-backed regex or bind to
  tiktoken's Rust core.
- **AMP is autocast-only, no `GradScaler`.** LibTorch's C++ AMP support
  doesn't include a public, stable `GradScaler` equivalent across versions.
  `train --amp` wraps the forward pass in `at::autocast` on CUDA but skips
  dynamic loss scaling. Training in fp32 (the default) is unaffected and
  matches the Python script's numerics most closely; `--amp` trades a bit
  of that safety margin for speed/memory, same tradeoff the Python
  `GradScaler` was there to manage, just without the extra guardrail.
- **No peak-VRAM logging.** `torch.cuda.max_memory_allocated()` doesn't have
  a stable public LibTorch C++ equivalent, so that line from `model.py`'s
  logging is omitted. Step/loss/lr/tokens-per-sec logging is unchanged.
- **Weight tying is implemented differently, on purpose.** `model.py` does
  `self.lm_head.weight = self.wte.weight`, which works in Python because
  `nn.Module.__setattr__` re-registers the parameter. The literal C++
  equivalent (`lm_head->weight = wte->weight;`) does *not* do this — it
  leaves a second, silently-ungrounded parameter registered under
  `lm_head`, so gradients wouldn't actually accumulate correctly on both
  paths. Instead, `GPTImpl::forward` computes the output projection as
  `x @ wte.weight^T` directly, with no separate `lm_head` module at all.
  This gives true single-tensor weight tying, just structured differently.
- **Checkpoints are C++-only.** `torch::save`/`torch::load` use a different
  archive format than Python's `torch.save`/`torch.load`, so
  `TinyStoriesV2-GPT4-model.pt` written by `train` here isn't cross-loadable
  with your original Python checkpoint (or vice versa). Everything stays
  self-consistent within this C++ project.
- **`BinDataset` uses `mmap` directly (POSIX)**, matching the intent of
  `np.memmap` in `model.py` (don't load the whole token file into RAM).
  Linux/macOS only as written; Windows would need
  `CreateFileMapping`/`MapViewOfFile` in its place.
- **`generate`'s sampling reproduces `test.py` exactly, including its
  quirk**: it scales logits by `temperature` and restricts to top-k, but
  then picks via `argmax` rather than multinomial sampling. Since
  softmax/argmax is monotonic in the logits, `temperature` doesn't actually
  change which token gets picked in that script — only `top_k` does. This
  port keeps that behavior rather than "fixing" it, since you asked for the
  same configs and everything; happy to switch it to true multinomial
  sampling if you'd rather it use temperature for real.
