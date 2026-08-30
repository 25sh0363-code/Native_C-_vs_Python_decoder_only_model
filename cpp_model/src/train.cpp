#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include "gpt_config.h"
#include "gpt_model.h"
#include "bin_dataset.h"

// Port of get_lr() in model.py — linear warmup then cosine decay to 0.
double get_lr(int64_t step, const TrainConfig& tc) {
    if (step < tc.warmup_steps) {
        return tc.learning_rate * (double)step / (double)tc.warmup_steps;
    }
    double progress = (double)(step - tc.warmup_steps) / (double)(tc.max_steps - tc.warmup_steps);
    return tc.learning_rate * 0.5 * (1.0 + std::cos(M_PI * progress));
}

// Port of estimate_loss() in model.py — averages loss over eval_iters
// validation batches with the model in eval mode (dropout off).
double estimate_loss(GPT& model, BinDataset& valid_data, const TrainConfig& tc, torch::Device device) {
    model->eval();
    torch::NoGradGuard no_grad;
    double total = 0.0;
    for (int64_t i = 0; i < tc.eval_iters; ++i) {
        auto [xb, yb] = valid_data.get_batch(tc.batch_size, device);
        auto [logits, loss] = model->forward(xb, yb);
        total += loss.item<double>();
    }
    model->train();
    return total / (double)tc.eval_iters;
}

// Port of save_checkpoint(step) in model.py. Python saves a single dict
// with model_state_dict, optimizer_state_dict, step, and config; the
// libtorch equivalent is a single archive with nested sub-archives so it
// stays one file, matching that structure rather than three separate files.
void save_checkpoint(GPT& model, torch::optim::AdamW& optimizer, int64_t step,
                      const GPTConfig& gcfg, const std::string& path) {
    torch::serialize::OutputArchive archive;

    torch::serialize::OutputArchive model_archive;
    model->save(model_archive);
    archive.write("model", model_archive);

    torch::serialize::OutputArchive optim_archive;
    optimizer.save(optim_archive);
    archive.write("optimizer", optim_archive);

    archive.write("step", torch::tensor(step));
    archive.write("vocab_size", torch::tensor(gcfg.vocab_size));
    archive.write("block_size", torch::tensor(gcfg.block_size));
    archive.write("n_layer", torch::tensor(gcfg.n_layer));
    archive.write("n_head", torch::tensor(gcfg.n_head));
    archive.write("n_embd", torch::tensor(gcfg.n_embd));

    archive.save_to(path);
}

int main() {
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using GPU (CUDA)\n";
    } else {
        std::cout << "Using CPU\n";
    }

    GPTConfig gcfg;
    TrainConfig tc;

    GPT model(gcfg);
    model->to(device);
    model->train();

    // Decay / no-decay param split, mirrors the loop over
    // model.named_parameters() in model.py.
    std::vector<torch::Tensor> decay_params, no_decay_params;
    for (const auto& p : model->named_parameters()) {
        if (!p.value().requires_grad()) continue;
        if (p.value().dim() < 2) {
            no_decay_params.push_back(p.value());
        } else {
            decay_params.push_back(p.value());
        }
    }

    auto decay_opts = std::make_unique<torch::optim::AdamWOptions>(tc.learning_rate);
    decay_opts->weight_decay(tc.weight_decay).betas({tc.beta1, tc.beta2});

    auto no_decay_opts = std::make_unique<torch::optim::AdamWOptions>(tc.learning_rate);
    no_decay_opts->weight_decay(0.0).betas({tc.beta1, tc.beta2});

    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.emplace_back(decay_params, std::move(decay_opts));
    groups.emplace_back(no_decay_params, std::move(no_decay_opts));

    torch::optim::AdamW optimizer(groups, torch::optim::AdamWOptions(tc.learning_rate));

    BinDataset train_data(tc.train_bin, gcfg.block_size);
    BinDataset valid_data(tc.valid_bin, gcfg.block_size);

    int64_t tokens_per_step = tc.batch_size * gcfg.block_size;
    auto start_time = std::chrono::steady_clock::now();

    for (int64_t step = 0; step < tc.max_steps; ++step) {
        double lr = get_lr(step, tc);
        // Base class virtual set_lr() — avoids downcasting each group's
        // options() to AdamWOptions& just to update the learning rate.
        for (auto& group : optimizer.param_groups()) {
            group.options().set_lr(lr);
        }

        auto [xb, yb] = train_data.get_batch(tc.batch_size, device);
        optimizer.zero_grad();

        auto [logits, loss] = model->forward(xb, yb);
        loss.backward();

        torch::nn::utils::clip_grad_norm_(model->parameters(), tc.grad_clip);
        optimizer.step();

        if (step % 100 == 0) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();
            double tokens_processed = (double)(step + 1) * tokens_per_step;
            double tok_per_sec = elapsed > 0.0 ? tokens_processed / elapsed : 0.0;

            std::cout << std::fixed << std::setprecision(4)
                      << "Step " << step
                      << ": loss = " << loss.item<double>()
                      << ", lr = " << lr
                      << std::setprecision(0)
                      << ", speed = " << tok_per_sec << " tok/s\n";
        }

        if (step % tc.eval_interval == 0 && step > 0) {
            double val_loss = estimate_loss(model, valid_data, tc, device);
            std::cout << "Step " << step << ": validation loss = " << val_loss << "\n";
            save_checkpoint(model, optimizer, step, gcfg, tc.ckpt_path);
        }
    }

    std::cout << "Training complete. Model saved to " << tc.ckpt_path << "\n";
    return 0;
}