#pragma once
#include <cstdint>
#include <string>

// Mirrors GPTConfig in model.py exactly. Keep these two in sync manually,
// or generate this header from the Python config to avoid drift.
struct GPTConfig {
    int64_t vocab_size = 50257;
    int64_t block_size = 256;
    int64_t n_layer    = 8;
    int64_t n_head     = 8;
    int64_t n_embd     = 384;
    double  dropout    = 0.1;
};

// Training hyperparameters — mirrors the top-level constants in model.py
struct TrainConfig {
    int64_t batch_size    = 32;
    double  learning_rate = 3e-4;
    double  weight_decay  = 0.1;
    double  beta1         = 0.9;
    double  beta2         = 0.95;
    double  grad_clip     = 1.0;

    int64_t max_steps     = 100000;
    int64_t warmup_steps  = 500;
    int64_t eval_interval = 1000;
    int64_t eval_iters    = 200;

    std::string train_bin = "data/train.bin";
    std::string valid_bin = "data/val.bin";
    std::string ckpt_path = "checkpoints/TinyStoriesV2-GPT4-model.pt";
};