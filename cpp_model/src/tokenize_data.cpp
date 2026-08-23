// C++ port of tokenizer.py:
// splits each source file on <|endoftext|>, BPE-encodes each story (with the
// EOT token appended), and streams the resulting token ids to a raw
// little-endian uint16 .bin file, exactly like np.uint16.tofile() did.

#include "bpe_tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string read_whole_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<std::string> split_on_eot(const std::string& data, const std::string& sep) {
    std::vector<std::string> stories;
    size_t start = 0;
    while (true) {
        size_t pos = data.find(sep, start);
        std::string piece = (pos == std::string::npos)
            ? data.substr(start)
            : data.substr(start, pos - start);

        // trim whitespace, matching python's str.strip()
        size_t a = piece.find_first_not_of(" \t\r\n");
        size_t b = piece.find_last_not_of(" \t\r\n");
        if (a != std::string::npos) {
            stories.push_back(piece.substr(a, b - a + 1));
        }

        if (pos == std::string::npos) break;
        start = pos + sep.size();
    }
    return stories;
}

void pretokenize_to_bin(const std::string& filename,
                         const std::string& output_filename,
                         const GPT2Tokenizer& tok) {
    std::ifstream check(filename);
    if (!check) {
        std::cout << "File " << filename << " does not exist. Skipping.\n";
        return;
    }
    check.close();

    std::string data = read_whole_file(filename);
    auto stories = split_on_eot(data, "<|endoftext|>");
    data.clear();
    data.shrink_to_fit();

    std::ofstream out(output_filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open output file " + output_filename);
    }

    uint64_t total_tokens = 0;
    for (const auto& story : stories) {
        auto ids = tok.encode(story + "<|endoftext|>");
        std::vector<uint16_t> buf(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            buf[i] = static_cast<uint16_t>(ids[i]);
        }
        out.write(reinterpret_cast<const char*>(buf.data()),
                   static_cast<std::streamsize>(buf.size() * sizeof(uint16_t)));
        total_tokens += buf.size();
    }

    std::cout << "Pre-tokenized " << filename << " to " << output_filename
              << ". Total tokens: " << total_tokens << "\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string encoder_json = "encoder.json";
    std::string vocab_bpe = "vocab.bpe";
    std::string train_file = "TinyStoriesV2-GPT4-train.txt";
    std::string valid_file = "TinyStoriesV2-GPT4-valid.txt";

    // optional overrides: tokenize_data [encoder.json] [vocab.bpe] [train.txt] [valid.txt]
    if (argc > 1) encoder_json = argv[1];
    if (argc > 2) vocab_bpe = argv[2];
    if (argc > 3) train_file = argv[3];
    if (argc > 4) valid_file = argv[4];

    GPT2Tokenizer tok(encoder_json, vocab_bpe);

    std::cout << "Pre-tokenizing training data...\n";
    pretokenize_to_bin(train_file, "train.bin", tok);

    std::cout << "Pre-tokenizing validation data...\n";
    pretokenize_to_bin(valid_file, "val.bin", tok);

    return 0;
}
