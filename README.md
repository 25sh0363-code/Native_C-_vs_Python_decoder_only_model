# Native C++ vs Python Decoder-Only Transformer

A comparative implementation and benchmarking project for a small GPT-style decoder-only Transformer written in:

- **Python + PyTorch**
- **C++ + LibTorch** (the PyTorch C++ API)

Both implementations train a language model from scratch on **TinyStories V2 GPT-4** using GPT-2-compatible token IDs. The project compares training throughput, autoregressive inference throughput, and recorded memory use on an NVIDIA RTX 4090.

> **Important:** This is a comparison of the tested Python/PyTorch and C++/LibTorch configurations. It is not a pure programming-language benchmark or a comparison against a framework-free handwritten CUDA implementation.

---

## Results at a glance

The recorded benchmarks were run on an NVIDIA RTX 4090.

| Metric | C++ / LibTorch | Python / PyTorch | Observed result |
|---|---:|---:|---|
| Training throughput | ~95,000–100,000 tok/s | ~59,000–61,000 tok/s | C++ was ~1.56×–1.69× faster |
| Representative training rate | ~97,500 tok/s | ~60,000 tok/s | C++ was ~1.63× faster |
| Median inference throughput | 190.6 tok/s | 129 tok/s | C++ was ~1.48× faster |
| Median recorded system RAM | ~864.7 MB | ~1,055 MB | C++ used ~18% less |
| Median recorded GPU memory | ~164.78 MB | ~545 MB | C++ used ~69.8% less |

The inference result is based on three recorded runs for the prompt:

```text
A small dog found a shiny ball
```

The models may generate different output lengths because generation stops when an end-of-text token is emitted. Therefore, **tokens/sec** is the primary inference metric in the current experiment; raw end-to-end generation time is not treated as a fixed-workload latency comparison.

---

## Research question

> How do Python/PyTorch and C++/LibTorch implementations of the same small decoder-only Transformer compare in training throughput, inference throughput, and memory use on an NVIDIA RTX 4090?

---

## Architecture

Both implementations use the same GPT-style decoder-only Transformer configuration.

| Hyperparameter | Value |
|---|---:|
| Model type | Decoder-only Transformer / GPT-style language model |
| Vocabulary size | 50,257 (GPT-2) |
| Context length | 256 tokens |
| Transformer layers | 8 |
| Attention heads | 8 |
| Embedding width | 384 |
| Attention head dimension | 48 |
| MLP hidden width | 1,536 |
| Position embeddings | Learned absolute position embeddings |
| Normalization | Pre-LayerNorm |
| Activation | GELU |
| Attention | Causal multi-head self-attention |
| Dropout | 0.1 |
| Input/output embedding weights | Tied |

The model performs next-token prediction. For an input sequence:

```text
[x1, x2, x3, ..., xT]
```

its targets are shifted by one token:

```text
[x2, x3, x4, ..., x(T+1)]
```

---

## Dataset and tokenization

The model is trained on **TinyStories V2 GPT-4**.

The original Python preprocessing script uses the GPT-2 tokenizer from `tiktoken`:

```python
enc = tiktoken.get_encoding("gpt2")
```

Text is tokenized and stored as raw `uint16` binary token streams:

```text
train.bin
val.bin
```

| File | Purpose | Format |
|---|---|---|
| `train.bin` | Training token stream | Raw `uint16` GPT-2 token IDs |
| `val.bin` | Validation token stream | Raw `uint16` GPT-2 token IDs |

The GPT-2 vocabulary has 50,257 token IDs, so it fits in an unsigned 16-bit integer. The GPT-2 end-of-text token is ID `50256`.

For the main experiment, the same pre-tokenized `train.bin` and `val.bin` files should be supplied to both implementations.

---

## Project structure

```text
Native_C-_vs_Python_decoder_only_model/
│
├── python_model/
│   ├── model.py                 
│   ├── test.py                  
│   ├── tokenizer.py             
│   ├── requirements.txt         
│   └──log files/
|        |_generation_output.txt
|        |_train.log
│
├── cpp_model/
│   ├── CMakeLists.txt           
│   ├── include/
│   │   ├── gpt_model.h          
│   │   ├── bin_dataset.h        
│   │   └── gpt_config.h      
│   ├── src/
│   │   ├── gpt_model.cpp       
│   │   ├── bin_dataset.cpp      
│   │   ├── train.cpp            
│   │   └── generate.cpp         
│   |__data/
|   |   |__.bin files generated using tokenizer.py
|   |__traincpp.log
|   |__tokenizer.py.  #same tokenizer used for python model
│
└── README.md
```


---

## Python setup

### Requirements

Install the Python dependencies:

```bash
cd python_model
python3 -m pip install -r requirements.txt
```

Typical dependencies:

```text
torch>=2.0.0
tiktoken>=0.4.0
numpy>=1.24.0
tqdm>=4.65.0
```

### Pre-tokenize TinyStories

Place the raw text files in the Python project directory, then run:

```bash
python3 tokenizer.py
```

This writes:

```text
train.bin
val.bin
```

### Train the Python model

```bash
python3 model.py
```

The Python implementation:

- Chooses CUDA when available.
- Uses AdamW.
- Uses a warmup-plus-cosine learning-rate schedule.
- Uses gradient clipping.
- Performs periodic validation.
- Saves a Python checkpoint.
- Uses CUDA AMP/autocast and `GradScaler` when CUDA is available.(slightly diffrent from cpp model which potentially might have affected the t/s during training)

### Generate with Python

```bash
python3 test.py
```

---

## C++ / LibTorch setup

The C++ implementation uses **LibTorch**, PyTorch's C++ API, for autograd, CUDA execution, optimizers, and tensor operations.

### Requirements

You need:

- CMake 3.18 or newer
- A C++17-compatible compiler
- NVIDIA GPU and CUDA-compatible LibTorch for GPU training
- The LibTorch package matching the cloud machine's CUDA environment
- Internet access during the first CMake configuration if `nlohmann/json` is fetched with CMake `FetchContent`

### Configure and build

Set `CMAKE_PREFIX_PATH` to the extracted LibTorch directory.

```bash
cd cpp_model

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch

cmake --build build --config Release -j2
```

The build should produce executables similar to:

```text
build/train
build/generate
build/tokenize_data
```

### Use existing token binaries

For a direct comparison, copy or symlink the same Python-generated token files into the C++ working directory:

```bash
cp ../python_model/train.bin .
cp ../python_model/val.bin .
```

### Train the C++ model

```bash
./build/train
```

The C++ trainer:

- Selects CUDA when available.
- Reads `train.bin` and `val.bin`.
- Samples contiguous next-token training windows.
- Uses AdamW.
- Uses warmup followed by cosine learning-rate decay.
- Clips gradient norms.
- Runs periodic validation.
- Saves a C++/LibTorch checkpoint.

### Generate with C++

If the C++ tokenizer files are required, place `encoder.json` and `vocab.bpe` where the program can access them.

```bash
./build/generate encoder.json vocab.bpe
```

The generation program loads the C++ checkpoint, tokenizes the prompt, generates autoregressively, prints text, calculates throughput, and saves a generation-output file.

---

## Training configuration

| Setting | Value |
|---|---:|
| Dataset | TinyStories V2 GPT-4 |
| Tokenizer family | GPT-2 byte-pair encoding |
| Batch size | 32 |
| Context length | 256 |
| Tokens per optimizer step | 8,192 |
| Training steps | 100,000 |
| Nominal training tokens | 819,200,000 |
| Objective | Next-token cross-entropy |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Adam betas | (0.9, 0.95) |
| Weight decay | 0.1 for matrix-like parameters |
| Weight decay for biases/norm | 0.0 |
| Gradient clipping | 1.0 |
| Warmup steps | 500 |
| LR schedule | Cosine decay after warmup |
| Validation interval | 1,000 steps |
| Validation iterations | 200 |

\[
32 \times 256 = 8{,}192 \text{ tokens per training step}
\]

\[
100{,}000 \times 8{,}192 = 819{,}200{,}000 \text{ nominal training tokens}
\]

---

## Benchmark results

### Training throughput

The observed sustained training throughput on an RTX 4090 was:

| Implementation | Observed training speed |
|---|---:|
| C++ / LibTorch | ~95,000–100,000 tok/s |
| Python / PyTorch | ~59,000–61,000 tok/s |

Using the midpoints of the observed ranges:

\[
\frac{97{,}500}{60{,}000}=1.625
\]

The C++ implementation achieved approximately **1.63×** Python's training throughput, or approximately **62.5% higher** throughput.

Using the full observed ranges, the C++ advantage was approximately **1.56× to 1.69×**.

### Inference throughput and memory

Prompt:

```text
A small dog found a shiny ball
```

Three inference measurements were recorded for each implementation.

| Metric | C++ Run 1 | C++ Run 2 | C++ Run 3 | Python Run 1 | Python Run 2 | Python Run 3 |
|---|---:|---:|---:|---:|---:|---:|
| Inference throughput | 190.6 tok/s | 190.6 tok/s | 192.254 tok/s | 129 tok/s | 123 tok/s | 130 tok/s |
| Generation time | 0.666 s | 0.666 s | 0.659 s | 0.61 s | 0.62 s | 0.61 s |
| Recorded system RAM | 866.098 MB | 864.7 MB | 864.844 MB | 1,055 MB | 1,055 MB | 1,055 MB |
| Recorded GPU memory | 164.78 MB | 164.78 MB | 164.79 MB | 545 MB | 545 MB | 545 MB |

Median summary:

| Metric | C++ / LibTorch | Python / PyTorch | Difference |
|---|---:|---:|---|
| Median inference throughput | **190.6 tok/s** | 129 tok/s | C++ is ~1.48× faster |
| Mean inference throughput | **191.15 tok/s** | 127.33 tok/s | C++ is ~1.50× faster |
| Median recorded system RAM | **864.7 MB** | 1,055 MB | C++ uses ~18.0% less |
| Median recorded GPU memory | **164.78 MB** | 545 MB | C++ uses ~69.8% less |

---

## Important limitations

This repository intentionally documents the following constraints so the results are interpreted correctly.

### Precision policy differs

- Python training used CUDA AMP/autocast and `GradScaler`.
- The recorded C++ benchmark trained in FP32.

The results therefore reflect the complete tested configurations, including their numerical precision policies. Do not attribute all performance differences solely to C++ versus Python.


### C++ tokenizer limitation

The C++ GPT-2 BPE tokenizer has an ASCII-oriented approximation of the full Unicode pre-tokenizer regex used by `tiktoken`. It is appropriate for English TinyStories-style text but can diverge on non-ASCII input such as accented text, CJK characters, or emoji.

### Sample outputs

-Python model:
    A small dog found a shiny ball. The dog wanted the ball. The dog tried to get the ball, but it was too high. The dog was sad.
    Then, a big bird came. The bird saw the dog and the shiny ball. The bird wanted to help. The bird flew up and got the ball for the dog. The dog was happy. The dog and the bird played with the shiny ball all day.

-C++ model:
    A small dog found a shiny ball. The dog wanted to play with the ball. The dog ran to the ball and gave it a big push. The ball rolled and rolled. The dog was happy.
    The dog saw a big tree. The dog wanted to play with the ball. The dog pushed the ball with his nose. The ball rolled and rolled. The dog chased the ball. The dog was very happy.
    But then, the ball rolled into a hole. The dog tried to get the ball out, but it was too deep. The dog could not get the ball out. The dog was sad too. The end.

---

## Recommended fair inference protocol

For a stronger follow-up benchmark, use this exact procedure in both implementations:

```text
Hardware: Same RTX 4090 machine
Prompt: Identical string and identical GPT-2 token IDs
Generation mode: Greedy argmax
New tokens: Exactly 256
EOT stopping: Disabled during timing
Warm-up runs: 3
Measured runs: 3 to 5 does the job
Timing: CUDA synchronize immediately before and after each run
Reported metric: Median generation time and median tokens/sec
Memory: Same tool and same measurement point
```

---

## Reproducibility checklist

Record these values for every final benchmark run:

- GPU model and GPU memory capacity
- CUDA version
- PyTorch version
- LibTorch version
- C++ compiler and version
- Operating system
- Exact model configuration
- Batch size and context length
- Exact tokenizer and input prompt
- Number of training steps and tokens processed
- Precision mode
- Learning-rate schedule and optimizer settings
- Inference max-token limit and EOT behavior
- Warm-up and measured-run counts
- Exact command used to launch the program

---

## Conclusion

This project demonstrates that a small decoder-only Transformer can be implemented, trained, and evaluated in both Python/PyTorch and C++/LibTorch using the same general architecture and TinyStories token-data pipeline.

In the recorded NVIDIA RTX 4090 experiments, the C++/LibTorch configuration achieved higher observed training throughput, higher observed autoregressive inference throughput, lower recorded system RAM use, and lower recorded GPU memory use than the Python/PyTorch configuration. The results are specific to the tested code, framework versions, hardware, tokenizer behavior, generation settings, and precision policies.

Future work should align numerical precision across implementations, use a fixed-length multi-run inference protocol, add key-value caching consistently, record standardized peak-memory measurements, and evaluate language-model quality with additional quantitative metrics.

---

## License

-none yet

---

## Acknowledgments

- Attention is all you need(2017)-https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- Andrej karpathy's lectrues on gpt2 recreation and decoder only transformers - https://youtu.be/kCc8FmEb1nY?si=5y37KxxpiBomNoQR and https://youtu.be/l8pRSuU81PU?si=2i59BjPM8fJaJtGE
- Basics of neural network in a transformer and implementations -Josh stamer https://www.youtube.com/@statquest
- Race Engineering for cloud GPU services https://raceengineering.ai