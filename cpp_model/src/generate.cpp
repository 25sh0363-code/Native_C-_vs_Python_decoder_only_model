#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <limits>
#include <sys/resource.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "gpt_config.h"
#include "gpt_model.h"

torch::Tensor generate(
    GPT& model,
    torch::Tensor idx,
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

    std::vector<int64_t> prompt_ids = {32, 1402, 3290, 1043, 257, 22441, 2613};
    auto idx = torch::tensor(prompt_ids, torch::kLong).unsqueeze(0).to(device);

    if (device.is_cuda()) {
        torch::cuda::synchronize();
        c10::cuda::CUDACachingAllocator::resetPeakStats(0);
    }
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

    std::cout << "Generated token ids: ";
    for (int64_t i = 0; i < out.size(1); ++i) {
        std::cout << out[0][i].item<int64_t>() << " ";
    }
    std::cout << "\n";

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Peak RAM usage: " << (usage.ru_maxrss / 1024.0) << " MB\n";

    if (device.is_cuda()) {
        auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
        double peak_vram_mb = stats.allocated_bytes[0].peak / (1024.0 * 1024.0);
        std::cout << "Peak GPU memory usage: " << peak_vram_mb << " MB\n";
    }

    return 0;
}