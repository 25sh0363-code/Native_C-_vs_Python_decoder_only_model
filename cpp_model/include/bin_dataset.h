#pragma once
// Mirrors Bindataset in model.py: memory-maps a raw uint16 token file
// and samples random contiguous blocks for (x, y) next-token pairs.
//
// POSIX only (Linux / macOS) since it uses mmap directly, same as the
// intent behind np.memmap in the original (don't load the whole file
// into RAM). For Windows, swap mmap/munmap for CreateFileMapping/MapViewOfFile.

#include <torch/torch.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

class BinDataset {
public:
    BinDataset(const std::string& path, int64_t block_size, torch::Device device)
        : block_size_(block_size), device_(device) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("BinDataset: failed to open " + path);
        }
        struct stat st{};
        if (::fstat(fd_, &st) != 0) {
            ::close(fd_);
            throw std::runtime_error("BinDataset: fstat failed for " + path);
        }
        file_size_ = static_cast<size_t>(st.st_size);
        num_tokens_ = static_cast<int64_t>(file_size_ / sizeof(uint16_t));

        void* mapped = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd_);
            throw std::runtime_error("BinDataset: mmap failed for " + path);
        }
        data_ = static_cast<const uint16_t*>(mapped);
    }

    ~BinDataset() {
        if (data_ != nullptr) {
            ::munmap(const_cast<uint16_t*>(data_), file_size_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    BinDataset(const BinDataset&) = delete;
    BinDataset& operator=(const BinDataset&) = delete;

    int64_t num_tokens() const { return num_tokens_; }

    std::pair<torch::Tensor, torch::Tensor> get_batch(int64_t batch_size) {
        const int64_t max_start = num_tokens_ - block_size_ - 1;
        if (max_start < 0) {
            throw std::runtime_error("BinDataset: file smaller than block_size + 1");
        }

        std::uniform_int_distribution<int64_t> dist(0, max_start);

        auto x = torch::empty({batch_size, block_size_}, torch::kInt64);
        auto y = torch::empty({batch_size, block_size_}, torch::kInt64);
        auto x_acc = x.accessor<int64_t, 2>();
        auto y_acc = y.accessor<int64_t, 2>();

        for (int64_t b = 0; b < batch_size; ++b) {
            const int64_t start = dist(rng_);
            for (int64_t t = 0; t < block_size_; ++t) {
                x_acc[b][t] = static_cast<int64_t>(data_[start + t]);
                y_acc[b][t] = static_cast<int64_t>(data_[start + t + 1]);
            }
        }

        return {x.to(device_), y.to(device_)};
    }

private:
    int fd_ = -1;
    const uint16_t* data_ = nullptr;
    size_t file_size_ = 0;
    int64_t num_tokens_ = 0;
    int64_t block_size_;
    torch::Device device_;
    std::mt19937_64 rng_{std::random_device{}()};
};
