#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <limits>
#include "gpt_config.h"
#include "gpt_model.h"

// ---------------------------------------------------------------------
// TOKENIZATION GAP: this file assumes you already have prompt token ids
// as a std::vector<int64_t>. GPT-2 BPE encode/decode is not implemented
// here. Two options:
//   1. Keep tokenization in Python: encode the prompt with tiktoken,
//      pass the ids in (e.g. via a small file or argv), run only this
//      generation loop natively, decode the output ids back in Python.
//   2. Port a real BPE encoder/decoder (~150 lines) using an exported
//      vocab.bpe/encoder.json, for fully native end-to-end inference.
// Which one you need depends on whether your paper's benchmark scope is
// "training loop only" or "full native pipeline including tokenization."
// ---------------------------------------------------------------------

// Port of generate() in test.py.
torch::Tensor generate(
    GPT& model,
    torch::Tensor idx,           // [1, T] token ids
    int64_t max_new_tokens,
    double temperature,
    int64_t top_k,
    const GPTConfig& cfg,
    int64_t eot_token
) {
    model->eval();
    torch::NoGradGuard no_grad;

    for (int64_t i = 0; i < max_new_tokens; ++i) {
        int64_t T = idx.size(1);
        auto idx_cond = T <= cfg.block_size
            ? idx
            : idx.slice(/*dim=*/1, T - cfg.block_size, T);

        auto [logits, unused_loss] = model->forward(idx_cond);
        (void)unused_loss;

        auto last_logits = logits.index(
            {torch::indexing::Slice(), -1, torch::indexing::Slice()}
        ) / temperature;

        auto topk_vals = std::get<0>(last_logits.topk(top_k, /*dim=*/-1));
        auto threshold = topk_vals.index({torch::indexing::Slice(), -1}).unsqueeze(-1);
        last_logits = last_logits.masked_fill(
            last_logits < threshold, -std::numeric_limits<float>::infinity());

        auto probs = torch::softmax(last_logits, -1);
        auto next_token = probs.argmax(-1, /*keepdim=*/true);

        if (next_token.item<int64_t>() == eot_token) break;
        idx = torch::cat({idx, next_token}, /*dim=*/1);
    }

    return idx;
}

// Loads a checkpoint written by train.cpp's save_checkpoint(): a nested
// archive with "model", "optimizer", "step" sub-archives. Only "model"
// is needed for inference.
void load_model_checkpoint(GPT& model, const std::string& path) {
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    torch::serialize::InputArchive model_archive;
    archive.read("model", model_archive);
    model->load(model_archive);
}

int main() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) device = torch::kCUDA;

    GPTConfig cfg;
    GPT model(cfg);

    load_model_checkpoint(model, "checkpoints/TinyStoriesV2-GPT4-model.pt");
    model->to(device);
    model->eval();

    std::cout << "Model loaded successfully.\n";

    // Placeholder prompt ids — see tokenization gap note above.
    // Replace with real BPE-encoded ids for "A small dog found a shiny blue ball"
    std::vector<int64_t> prompt_ids = {32, 1402, 3290}; // TODO: real encode()
    auto idx = torch::tensor(prompt_ids, torch::kLong).unsqueeze(0).to(device);

    if (device.is_cuda()) torch::cuda::synchronize();
    auto start = std::chrono::steady_clock::now();

    auto out = generate(
        model, idx,
        /*max_new_tokens=*/256, /*temperature=*/0.4, /*top_k=*/50,
        cfg, /*eot_token=*/50256
    );

    if (device.is_cuda()) torch::cuda::synchronize();
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    int64_t num_generated = out.size(1) - (int64_t)prompt_ids.size();
    double tok_per_sec = elapsed > 0.0 ? num_generated / elapsed : 0.0;

    std::cout << "Generated " << num_generated << " tokens in " << elapsed
              << " sec (" << tok_per_sec << " tok/s)\n";

    // TODO: decode(out) back to text once a BPE decoder is wired in.
    return 0;
}