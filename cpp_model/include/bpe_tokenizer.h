#pragma once
// A from-scratch C++ re-implementation of the GPT-2 byte-level BPE encoding
// that tiktoken.get_encoding("gpt2") uses in tokenizer.py / test.py.
//
// tiktoken itself is a Python/Rust library with no official C++ API, so this
// reproduces the underlying algorithm directly from OpenAI's public GPT-2
// encoder.json / vocab.bpe files:
//   https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json
//   https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe
//
// CAVEAT: the real GPT-2 pretokenizer regex relies on Unicode property
// escapes (\p{L}, \p{N}) that std::regex does not support. This uses an
// ASCII-letter/digit approximation instead. For plain English text (like
// TinyStories) the output is byte-for-byte identical to tiktoken; it only
// diverges on non-ASCII input. See README.md for details.

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class GPT2Tokenizer {
public:
    GPT2Tokenizer(const std::string& encoder_json_path, const std::string& vocab_bpe_path);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t eot_token() const { return eot_token_id_; }

private:
    std::unordered_map<std::string, int32_t> encoder_;   // byte-encoded token -> id
    std::unordered_map<int32_t, std::string> decoder_;   // id -> byte-encoded token
    std::unordered_map<std::string, int32_t> bpe_ranks_; // "tokA tokB" -> merge rank

    std::array<std::string, 256> byte_encoder_;           // raw byte -> unicode string
    std::unordered_map<std::string, uint8_t> byte_decoder_; // unicode string -> raw byte

    int32_t eot_token_id_ = -1;

    void build_byte_encoder();
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::string bpe_merge(const std::string& byte_encoded_token) const;
};
