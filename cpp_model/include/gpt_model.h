#pragma once
// C++ / LibTorch port of the GPT architecture in model.py.
// Same hyperparameters, same layer structure, same weight tying.

#include <torch/torch.h>
#include <cmath>
#include <utility>

struct GPTConfig {
    int64_t vocab_size = 50257;
    int64_t block_size = 256;
    int64_t n_layer    = 8;
    int64_t n_head     = 8;
    int64_t n_embd     = 384;
    double  dropout    = 0.1;
};

// ---------------------------------------------------------------------------
// CausalSelfAttention
// ---------------------------------------------------------------------------
struct CausalSelfAttentionImpl : torch::nn::Module {
    explicit CausalSelfAttentionImpl(const GPTConfig& config);
    torch::Tensor forward(torch::Tensor x);

    int64_t n_head;
    double dropout_p;
    torch::nn::Linear c_attn{nullptr};
    torch::nn::Linear c_proj{nullptr};
    torch::Tensor bias; // causal mask buffer, registered so it moves with .to(device)
};
TORCH_MODULE(CausalSelfAttention);

// ---------------------------------------------------------------------------
// Block (attention + MLP with residual connections)
// ---------------------------------------------------------------------------
struct BlockImpl : torch::nn::Module {
    explicit BlockImpl(const GPTConfig& config);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::LayerNorm ln1{nullptr};
    CausalSelfAttention attn{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    torch::nn::Sequential mlp{nullptr};
};
TORCH_MODULE(Block);

// ---------------------------------------------------------------------------
// GPT
// ---------------------------------------------------------------------------
struct GPTImpl : torch::nn::Module {
    explicit GPTImpl(const GPTConfig& config);

    // Returns {logits, loss}. loss is an undefined Tensor if targets is nullopt.
    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor idx,
        c10::optional<torch::Tensor> targets = c10::nullopt);

    GPTConfig config;
    torch::nn::Embedding wte{nullptr};
    torch::nn::Embedding wpe{nullptr};
    torch::nn::Dropout drop{nullptr};
    torch::nn::ModuleList h{nullptr};
    torch::nn::LayerNorm ln_f{nullptr};
    // NOTE: there is no separate lm_head module. The output projection reuses
    // wte->weight directly (logits = x @ wte.weight^T), which is what true
    // weight tying requires: a single shared Tensor/gradient, not two
    // separately-registered parameters that happen to start out equal.
    // (Re-pointing a second nn::Linear's `weight` member at wte->weight,
    // as a naive port of model.py's `self.lm_head.weight = self.wte.weight`
    // would do in LibTorch, leaves a stale, ungrounded duplicate parameter
    // registered under lm_head - it silently receives no gradient instead of
    // truly sharing one. Using wte->weight directly avoids that trap.)
};
TORCH_MODULE(GPT);
