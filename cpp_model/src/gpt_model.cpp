#include "gpt_model.h"
#include <limits>

// ---------------------------------------------------------------------------
// CausalSelfAttention
// ---------------------------------------------------------------------------
CausalSelfAttentionImpl::CausalSelfAttentionImpl(const GPTConfig& config)
    : n_head(config.n_head), dropout_p(config.dropout) {
    c_attn = register_module("c_attn", torch::nn::Linear(config.n_embd, 3 * config.n_embd));
    c_proj = register_module("c_proj", torch::nn::Linear(config.n_embd, config.n_embd));

    auto tril = torch::tril(torch::ones({config.block_size, config.block_size}));
    bias = tril.view({1, 1, config.block_size, config.block_size});
    register_buffer("bias", bias);
}

torch::Tensor CausalSelfAttentionImpl::forward(torch::Tensor x) {
    const auto sizes = x.sizes();
    const int64_t B = sizes[0], T = sizes[1], C = sizes[2];

    auto qkv = c_attn->forward(x); // [B, T, 3C]
    auto chunks = qkv.split(C, /*dim=*/2);
    auto q = chunks[0], k = chunks[1], v = chunks[2];

    q = q.view({B, T, n_head, C / n_head}).transpose(1, 2);
    k = k.view({B, T, n_head, C / n_head}).transpose(1, 2);
    v = v.view({B, T, n_head, C / n_head}).transpose(1, 2);

    auto att = torch::matmul(q, k.transpose(-2, -1)) *
               (1.0 / std::sqrt(static_cast<double>(C / n_head)));

    using torch::indexing::Slice;
    auto mask = bias.index({Slice(), Slice(), Slice(0, T), Slice(0, T)});
    att = att.masked_fill(mask == 0, -std::numeric_limits<float>::infinity());
    att = torch::softmax(att, -1);
    att = torch::dropout(att, dropout_p, is_training());

    auto y = torch::matmul(att, v);
    y = y.transpose(1, 2).contiguous().view({B, T, C});

    return c_proj->forward(y);
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------
BlockImpl::BlockImpl(const GPTConfig& config) {
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.n_embd})));
    attn = register_module("attn", CausalSelfAttention(config));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.n_embd})));

    mlp = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(config.n_embd, 4 * config.n_embd),
        torch::nn::GELU(),
        torch::nn::Linear(4 * config.n_embd, config.n_embd),
        torch::nn::Dropout(torch::nn::DropoutOptions(config.dropout))
    ));
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
    x = x + attn->forward(ln1->forward(x));
    x = x + mlp->forward(ln2->forward(x));
    return x;
}

// ---------------------------------------------------------------------------
// GPT
// ---------------------------------------------------------------------------
GPTImpl::GPTImpl(const GPTConfig& config_) : config(config_) {
    wte = register_module("wte", torch::nn::Embedding(config.vocab_size, config.n_embd));
    wpe = register_module("wpe", torch::nn::Embedding(config.block_size, config.n_embd));
    drop = register_module("drop", torch::nn::Dropout(torch::nn::DropoutOptions(config.dropout)));

    h = register_module("h", torch::nn::ModuleList());
    for (int64_t i = 0; i < config.n_layer; ++i) {
        h->push_back(Block(config));
    }

    ln_f = register_module("ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.n_embd})));
}

std::pair<torch::Tensor, torch::Tensor> GPTImpl::forward(
        torch::Tensor idx, c10::optional<torch::Tensor> targets) {
    const auto sizes = idx.sizes();
    const int64_t T = sizes[1];

    auto pos = torch::arange(T, idx.options().dtype(torch::kLong)).unsqueeze(0);

    auto x = drop->forward(wte->forward(idx) + wpe->forward(pos));
    for (size_t i = 0; i < h->size(); ++i) {
        x = h->ptr<BlockImpl>(i)->forward(x);
    }
    x = ln_f->forward(x);

    // Weight-tied output projection: logits = x @ wte.weight^T  (no bias,
    // matching nn.Linear(n_embd, vocab_size, bias=False) in model.py)
    auto logits = torch::matmul(x, wte->weight.transpose(0, 1));

    torch::Tensor loss;
    if (targets.has_value()) {
        auto logits_flat = logits.view({-1, logits.size(-1)});
        auto targets_flat = targets.value().view({-1});
        loss = torch::nn::functional::cross_entropy(logits_flat, targets_flat);
    }

    return {logits, loss};
}
