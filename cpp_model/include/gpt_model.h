#pragma once
#include <torch/torch.h>
#include <utility>
#include "gpt_config.h"

// --- CausalSelfAttention -----------------------------------------------
// Direct port of CausalSelfAttention in model.py.
struct CausalSelfAttentionImpl : torch::nn::Module {
    explicit CausalSelfAttentionImpl(const GPTConfig& cfg);

    torch::Tensor forward(torch::Tensor x);

    int64_t n_head;
    int64_t n_embd;
    double dropout_p;

    torch::nn::Linear c_attn{nullptr};
    torch::nn::Linear c_proj{nullptr};
    torch::Tensor bias; // causal mask buffer, not a trainable parameter
};
TORCH_MODULE(CausalSelfAttention);

// --- Block ---------------------------------------------------------------
// Direct port of Block in model.py.
struct BlockImpl : torch::nn::Module {
    explicit BlockImpl(const GPTConfig& cfg);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::LayerNorm ln1{nullptr};
    CausalSelfAttention attn{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    torch::nn::Sequential mlp{nullptr};
};
TORCH_MODULE(Block);

// --- GPT -------------------------------------------------------------------
// Direct port of GPT in model.py.
//
// Weight tying: unlike Python (`self.lm_head.weight = self.wte.weight`,
// which works because nn.Module.__setattr__ keeps _parameters in sync),
// libtorch has no such hook — reassigning a registered module's `weight`
// member after construction leaves a stale, separately-optimized duplicate
// in its parameter map. So there is no separate lm_head module here at all;
// the output projection reuses wte.weight directly in forward(), which is
// the only way to get genuine single-tensor weight tying in libtorch.
struct GPTImpl : torch::nn::Module {
    explicit GPTImpl(const GPTConfig& cfg);

    // Mirrors forward(idx, targets=None) -> (logits, loss)
    std::pair<torch::Tensor, torch::Tensor> forward(
        torch::Tensor idx,
        torch::optional<torch::Tensor> targets = torch::nullopt
    );

    GPTConfig config;

    torch::nn::Embedding wte{nullptr};
    torch::nn::Embedding wpe{nullptr};
    torch::nn::Dropout drop{nullptr};
    torch::nn::ModuleList h{nullptr};
    torch::nn::LayerNorm ln_f{nullptr};
    // No lm_head module — see comment above. Output projection is computed
    // in forward() via torch::nn::functional::linear(x, wte->weight).
};
TORCH_MODULE(GPT);