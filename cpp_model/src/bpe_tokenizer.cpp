#include "bpe_tokenizer.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

// --- UTF-8 helpers ---------------------------------------------------------

std::string utf8_encode_codepoint(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// Splits a UTF-8 string into a vector of single-codepoint UTF-8 strings.
std::vector<std::string> utf8_split_codepoints(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        len = std::min(len, s.size() - i);
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    for (size_t i = 0; i + 1 < word.size(); ++i) {
        pairs.insert({word[i], word[i + 1]});
    }
    return pairs;
}

} // namespace

void GPT2Tokenizer::build_byte_encoder() {
    // Replicates GPT-2's bytes_to_unicode(): printable ASCII / Latin-1 bytes
    // map to themselves; every other byte gets pushed into the private-use
    // range starting at 256, so every byte has a distinct, printable
    // unicode representation.
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    std::set<int> bs_set(bs.begin(), bs.end());

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (bs_set.find(b) == bs_set.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        std::string u = utf8_encode_codepoint(static_cast<uint32_t>(cs[i]));
        byte_encoder_[static_cast<uint8_t>(bs[i])] = u;
        byte_decoder_[u] = static_cast<uint8_t>(bs[i]);
    }
}

GPT2Tokenizer::GPT2Tokenizer(const std::string& encoder_json_path,
                              const std::string& vocab_bpe_path) {
    build_byte_encoder();

    // --- load encoder.json (token string -> id) ---
    std::ifstream enc_file(encoder_json_path);
    if (!enc_file) {
        throw std::runtime_error("GPT2Tokenizer: could not open " + encoder_json_path);
    }
    nlohmann::json j;
    enc_file >> j;
    for (auto it = j.begin(); it != j.end(); ++it) {
        int32_t id = it.value().get<int32_t>();
        encoder_[it.key()] = id;
        decoder_[id] = it.key();
    }

    auto eot_it = encoder_.find("<|endoftext|>");
    if (eot_it == encoder_.end()) {
        throw std::runtime_error("GPT2Tokenizer: <|endoftext|> not found in encoder.json");
    }
    eot_token_id_ = eot_it->second;

    // --- load vocab.bpe (merge list, priority = line order) ---
    std::ifstream bpe_file(vocab_bpe_path);
    if (!bpe_file) {
        throw std::runtime_error("GPT2Tokenizer: could not open " + vocab_bpe_path);
    }
    std::string line;
    std::getline(bpe_file, line); // skip the "#version..." header line
    int32_t rank = 0;
    while (std::getline(bpe_file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string a, b;
        iss >> a >> b;
        if (a.empty() || b.empty()) continue;
        bpe_ranks_[a + " " + b] = rank++;
    }
}

std::vector<std::string> GPT2Tokenizer::pretokenize(const std::string& text) const {
    // ASCII approximation of GPT-2's regex:
    //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    static const std::regex pattern(
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+)");

    std::vector<std::string> chunks;
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        chunks.push_back(it->str());
    }
    return chunks;
}

std::string GPT2Tokenizer::bpe_merge(const std::string& byte_encoded_token) const {
    std::vector<std::string> word = utf8_split_codepoints(byte_encoded_token);
    if (word.size() < 2) return byte_encoded_token;

    auto pairs = get_pairs(word);

    while (!pairs.empty()) {
        // find the pair with the lowest merge rank
        const std::pair<std::string, std::string>* best = nullptr;
        int32_t best_rank = INT32_MAX;
        for (const auto& p : pairs) {
            auto it = bpe_ranks_.find(p.first + " " + p.second);
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = &p;
            }
        }
        if (best == nullptr) break; // no more applicable merges

        const std::string first = best->first;
        const std::string second = best->second;

        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            // find next occurrence of `first` starting at i
            size_t j = i;
            while (j < word.size() && word[j] != first) ++j;
            new_word.insert(new_word.end(), word.begin() + i, word.begin() + j);
            i = j;
            if (i >= word.size()) break;

            if (word[i] == first && i + 1 < word.size() && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }

        word = std::move(new_word);
        if (word.size() < 2) break;
        pairs = get_pairs(word);
    }

    std::string result;
    for (size_t i = 0; i < word.size(); ++i) {
        if (i > 0) result += ' ';
        result += word[i];
    }
    return result;
}

std::vector<int32_t> GPT2Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    for (const auto& chunk : pretokenize(text)) {
        // map every raw byte of this chunk to its unicode surrogate
        std::string byte_encoded;
        byte_encoded.reserve(chunk.size() * 2);
        for (unsigned char c : chunk) {
            byte_encoded += byte_encoder_[c];
        }

        std::string merged = bpe_merge(byte_encoded);

        std::istringstream iss(merged);
        std::string tok;
        while (iss >> tok) {
            auto it = encoder_.find(tok);
            if (it != encoder_.end()) {
                ids.push_back(it->second);
            } else {
                // Should not happen with a complete vocab, but fail soft
                // rather than crash on unexpected input.
                for (const auto& cp : utf8_split_codepoints(tok)) {
                    auto it2 = encoder_.find(cp);
                    if (it2 != encoder_.end()) ids.push_back(it2->second);
                }
            }
        }
    }
    return ids;
}

std::string GPT2Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string byte_encoded;
    for (int32_t id : ids) {
        auto it = decoder_.find(id);
        if (it != decoder_.end()) byte_encoded += it->second;
    }

    std::string out;
    out.reserve(byte_encoded.size());
    for (const auto& cp : utf8_split_codepoints(byte_encoded)) {
        auto it = byte_decoder_.find(cp);
        if (it != byte_decoder_.end()) {
            out.push_back(static_cast<char>(it->second));
        }
    }
    return out;
}
