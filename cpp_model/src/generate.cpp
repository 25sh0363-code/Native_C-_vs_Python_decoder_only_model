// C++ / LibTorch port of test.py: loads a checkpoint, generates text
// autoregressively with temperature + top-k filtering, times it, and
// writes results to generation_output.txt.
//
// NOTE (faithfully ported from test.py, not a bug introduced here): the
// original script scales logits by temperature and restricts to the top-k
// logits, but then samples via argmax rather than multinomial sampling.
// Since softmax/argmax is monotonic in the logits, temperature has no
// effect on which token is picked - only top_k does. This port reproduces
// that exact behavior for fidelity.

#include "gpt_model.h"
#include "bpe_tokenizer.h"

#include <torch/torch.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

const std::string checkpoint_path = "TinyStoriesV2-GPT4-model.pt";

std::pair<std::string, int64_t> generate(
        GPT& model,
        const GPT2Tokenizer& tok,
        torch::Device device,
        const std::string& prompt,
        int64_t max_new_tokens = 256,
        double temperature = 0.4,
        int64_t top_k = 50) {
    model->eval();
    torch::NoGradGuard no_grad;

    auto ids = tok.encode(prompt);
    auto idx = torch::from_blob(ids.data(), {static_cast<int64_t>(ids.size())},
                                 torch::kInt32).to(torch::kLong).unsqueeze(0).to(device);
    int64_t start_len = idx.size(1);

    for (int64_t step = 0; step < max_new_tokens; ++step) {
        int64_t T = idx.size(1);
        int64_t from = std::max<int64_t>(0, T - model->config.block_size);
        auto idx_cond = idx.index({torch::indexing::Slice(),
                                     torch::indexing::Slice(from, T)});

        auto [logits_full, unused] = model->forward(idx_cond);
        auto logits = logits_full.index({torch::indexing::Slice(), -1, torch::indexing::Slice()})
                          / temperature;

        auto topk = torch::topk(logits, std::min(top_k, logits.size(-1)));
        auto values = std::get<0>(topk);
        auto min_keep = values.index({torch::indexing::Slice(), -1}).unsqueeze(-1);
        logits = logits.masked_fill(logits < min_keep, -std::numeric_limits<float>::infinity());

        auto probs = torch::softmax(logits, -1);
        auto next_token = std::get<1>(torch::max(probs, /*dim=*/-1, /*keepdim=*/true));

        if (next_token.item<int64_t>() == tok.eot_token()) break;
        idx = torch::cat({idx, next_token}, 1);
    }

    int64_t num_generated = idx.size(1) - start_len;
    auto idx_cpu = idx.to(torch::kCPU).squeeze(0);
    std::vector<int32_t> out_ids(idx_cpu.data_ptr<int64_t>(),
                                   idx_cpu.data_ptr<int64_t>() + idx_cpu.size(0));
    return {tok.decode(out_ids), num_generated};
}

} // namespace

int main(int argc, char** argv) {
    std::string encoder_json = "encoder.json";
    std::string vocab_bpe = "vocab.bpe";
    if (argc > 1) encoder_json = argv[1];
    if (argc > 2) vocab_bpe = argv[2];

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
    }
    std::cout << "Using device: " << (device.is_cuda() ? "cuda" : "cpu") << "\n";

    GPTConfig config{};
    GPT model(config);

    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint_path, device);
    model->load(archive);
    model->to(device);
    model->eval();

    std::cout << "Model loaded successfully.\n";

    GPT2Tokenizer tok(encoder_json, vocab_bpe);
    std::cout << "Model is ready for text generation.\n";

    std::string prompt = "A small dog found a shiny blue ball";

    if (device.is_cuda()) torch::cuda::synchronize();
    auto start = std::chrono::steady_clock::now();

    auto [output_text, num_generated] = generate(model, tok, device, prompt);

    if (device.is_cuda()) torch::cuda::synchronize();
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    double tok_per_sec = elapsed > 0 ? static_cast<double>(num_generated) / elapsed : 0.0;

    std::cout << output_text << "\n\n";
    std::cout << "Generated " << num_generated << " tokens in " << elapsed
              << " sec (" << tok_per_sec << " tok/s)\n";

    std::ofstream out("generation_output.txt");
    out << "Prompt: " << prompt << "\n\n";
    out << "Output:\n" << output_text << "\n\n";
    out << "Tokens generated: " << num_generated << "\n";
    out << "Time taken: " << elapsed << " sec\n";
    out << "Tokens/sec: " << tok_per_sec << "\n";

    std::cout << "Saved results to generation_output.txt\n";
    return 0;
}
