#include "gpt_model.h"
#include <cmath>
#include <limits>

// --- CausalSelfAttention -----------------------------------------------

CausalSelfAttentionImpl::CausalSelfAttentionImpl(const GPTConfig& cfg)
    : n_head(cfg.n_head), n_embd(cfg.n_embd), dropout_p(cfg.dropout) {

    c_attn = register_module("c_attn", torch::nn::Linear(cfg.n_embd, 3 * cfg.n_embd));
    c_proj = register_module("c_proj", torch::nn::Linear(cfg.n_embd, cfg.n_embd));

    auto mask = torch::tril(torch::ones({cfg.block_size, cfg.block_size}))
                    .view({1, 1, cfg.block_size, cfg.block_size});
    bias = register_buffer("bias", mask);
}

torch::Tensor CausalSelfAttentionImpl::forward(torch::Tensor x) {
    int64_t B = x.size(0), T = x.size(1), C = x.size(2);

    auto qkv = c_attn->forward(x).split(C, /*dim=*/2);
    auto q = qkv[0], k = qkv[1], v = qkv[2];

    q = q.view({B, T, n_head, C / n_head}).transpose(1, 2);
    k = k.view({B, T, n_head, C / n_head}).transpose(1, 2);
    v = v.view({B, T, n_head, C / n_head}).transpose(1, 2);

    auto att = torch::matmul(q, k.transpose(-2, -1)) * (1.0 / std::sqrt((double)k.size(-1)));

    auto causal_slice = bias.slice(/*dim=*/2, 0, T).slice(/*dim=*/3, 0, T);
    att = att.masked_fill(causal_slice == 0, -std::numeric_limits<float>::infinity());
    att = torch::softmax(att, /*dim=*/-1);
    att = torch::dropout(att, dropout_p, is_training());

    auto y = torch::matmul(att, v);
    y = y.transpose(1, 2).contiguous().view({B, T, C});

    return c_proj->forward(y);
}

// --- Block ---------------------------------------------------------------

BlockImpl::BlockImpl(const GPTConfig& cfg) {
    ln1 = register_module("ln1", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({cfg.n_embd})));
    attn = register_module("attn", CausalSelfAttention(cfg));
    ln2 = register_module("ln2", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({cfg.n_embd})));

    mlp = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(cfg.n_embd, 4 * cfg.n_embd),
        torch::nn::GELU(),
        torch::nn::Linear(4 * cfg.n_embd, cfg.n_embd),
        torch::nn::Dropout(cfg.dropout)
    ));
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
    x = x + attn->forward(ln1->forward(x));
    x = x + mlp->forward(ln2->forward(x));
    return x;
}

// --- GPT -------------------------------------------------------------------

GPTImpl::GPTImpl(const GPTConfig& cfg) : config(cfg) {
    wte = register_module("wte", torch::nn::Embedding(cfg.vocab_size, cfg.n_embd));
    wpe = register_module("wpe", torch::nn::Embedding(cfg.block_size, cfg.n_embd));
    drop = register_module("drop", torch::nn::Dropout(cfg.dropout));

    h = register_module("h", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layer; ++i) {
        h->push_back(Block(cfg));
    }

    ln_f = register_module("ln_f", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({cfg.n_embd})));

    // No lm_head module registered — see header comment on weight tying.
}

std::pair<torch::Tensor, torch::Tensor> GPTImpl::forward(
    torch::Tensor idx,
    torch::optional<torch::Tensor> targets
) {
    int64_t T = idx.size(1);
    auto pos = torch::arange(T, idx.options()).unsqueeze(0);

    auto x = drop->forward(wte->forward(idx) + wpe->forward(pos));
    for (const auto& module : *h) {
        x = module->as<Block>()->forward(x);
    }
    x = ln_f->forward(x);

    // Tied output projection: reuse wte.weight directly rather than a
    // separate lm_head module (see header comment).
    auto logits = torch::nn::functional::linear(x, wte->weight);

    torch::Tensor loss;
    if (targets.has_value()) {
        loss = torch::nn::functional::cross_entropy(
            logits.view({-1, logits.size(-1)}),
            targets->view({-1})
        );
    }

    return {logits, loss};
}