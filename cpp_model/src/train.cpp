// C++ / LibTorch port of the training loop in model.py.
// Same GPTConfig, same hyperparameters, same AdamW decay/no-decay param
// grouping, same cosine-with-warmup LR schedule, same checkpointing cadence.

#include "gpt_model.h"
#include "bin_dataset.h"

#include <torch/torch.h>
#include <ATen/autocast_mode.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

namespace {

// ---- config (mirrors model.py globals) ----
GPTConfig gpt_config{};

const int64_t batch_size = 32;
const double learning_rate = 3e-4;
const double weight_decay = 0.1;
const double beta1 = 0.9, beta2 = 0.95;
const double grad_clip = 1.0;

const int64_t max_steps = 100000;
const int64_t warmup_steps = 500;
const int64_t eval_interval = 1000;
const int64_t eval_iters = 200;

const std::string train_bin = "train.bin";
const std::string valid_bin = "val.bin";
const std::string checkpoint_path = "TinyStoriesV2-GPT4-model.pt";

double get_lr(int64_t step) {
    if (step < warmup_steps) {
        return learning_rate * static_cast<double>(step) / static_cast<double>(warmup_steps);
    }
    double progress = static_cast<double>(step - warmup_steps) /
                       static_cast<double>(max_steps - warmup_steps);
    return learning_rate * 0.5 * (1.0 + std::cos(M_PI * progress));
}

double estimate_loss(GPT& model, BinDataset& valid_data) {
    model->eval();
    torch::NoGradGuard no_grad;
    double total = 0.0;
    for (int64_t i = 0; i < eval_iters; ++i) {
        auto [xb, yb] = valid_data.get_batch(batch_size);
        auto [logits, loss] = model->forward(xb, yb);
        total += loss.item<double>();
    }
    model->train();
    return total / static_cast<double>(eval_iters);
}

void save_checkpoint(GPT& model, torch::optim::AdamW& optimizer, int64_t step) {
    torch::serialize::OutputArchive archive;
    model->save(archive);

    torch::serialize::OutputArchive optim_archive;
    optimizer.save(optim_archive);
    archive.write("optimizer", optim_archive);

    archive.write("step", torch::tensor({step}, torch::kInt64));

    archive.save_to(checkpoint_path);
}

} // namespace

int main(int argc, char** argv) {
    bool use_amp = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--amp") use_amp = true;
    }

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using GPU (CUDA)\n";
    } else {
        std::cout << "Using CPU\n";
    }

    GPT model(gpt_config);
    model->to(device);

    // decay / no-decay parameter grouping, same rule as model.py:
    // p.dim() < 2 -> no weight decay (biases, LayerNorm gains, ...)
    std::vector<torch::Tensor> decay_params, no_decay_params;
    for (auto& p : model->named_parameters()) {
        if (!p.value().requires_grad()) continue;
        if (p.value().dim() < 2) no_decay_params.push_back(p.value());
        else decay_params.push_back(p.value());
    }

    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.emplace_back(
        decay_params,
        std::make_unique<torch::optim::AdamWOptions>(
            torch::optim::AdamWOptions(learning_rate)
                .betas(std::make_tuple(beta1, beta2))
                .weight_decay(weight_decay)));
    groups.emplace_back(
        no_decay_params,
        std::make_unique<torch::optim::AdamWOptions>(
            torch::optim::AdamWOptions(learning_rate)
                .betas(std::make_tuple(beta1, beta2))
                .weight_decay(0.0)));

    torch::optim::AdamW optimizer(
        groups,
        torch::optim::AdamWOptions(learning_rate).betas(std::make_tuple(beta1, beta2)));

    BinDataset train_data(train_bin, gpt_config.block_size, device);
    BinDataset valid_data(valid_bin, gpt_config.block_size, device);

    model->train();

    const int64_t tokens_per_step = batch_size * gpt_config.block_size;
    auto start_time = std::chrono::steady_clock::now();

    for (int64_t step = 0; step < max_steps; ++step) {
        double lr = get_lr(step);
        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
        }

        auto [xb, yb] = train_data.get_batch(batch_size);
        optimizer.zero_grad();

        torch::Tensor logits, loss;
        if (use_amp && device.is_cuda()) {
            at::autocast::set_enabled(true);
            std::tie(logits, loss) = model->forward(xb, yb);
            at::autocast::clear_cache();
            at::autocast::set_enabled(false);
        } else {
            std::tie(logits, loss) = model->forward(xb, yb);
        }

        loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), grad_clip);
        optimizer.step();

        if (step % 100 == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
            double tokens_processed = static_cast<double>((step + 1) * tokens_per_step);
            double tok_per_sec = elapsed > 0 ? tokens_processed / elapsed : 0.0;

            std::cout << "Step " << step
                      << ": loss = " << loss.item<double>()
                      << ", lr = " << lr
                      << ", speed = " << tok_per_sec << " tok/s\n";
            // NOTE: peak-VRAM reporting (torch.cuda.max_memory_allocated() in
            // model.py) is omitted here - LibTorch does not expose a stable
            // public C++ API for that stat. See README.md.
        }

        if (step % eval_interval == 0 && step > 0) {
            double val_loss = estimate_loss(model, valid_data);
            std::cout << "Step " << step << ": validation loss = " << val_loss << "\n";
            save_checkpoint(model, optimizer, step);
        }
    }

    std::cout << "Training complete. Model saved to " << checkpoint_path << "\n";
    return 0;
}
