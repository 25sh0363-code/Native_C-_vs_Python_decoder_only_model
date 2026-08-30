#pragma once
#include <torch/torch.h>
#include <string>
#include <utility>
#include <cstdint>

// Port of Bindataset in model.py.
// Memory-maps a .bin file of uint16 token ids and samples random batches.
class BinDataset {
public:
    BinDataset(const std::string& path, int64_t block_size);
    ~BinDataset();

    // Owns a raw fd + mmap pointer — copying would double-close/double-munmap.
    BinDataset(const BinDataset&) = delete;
    BinDataset& operator=(const BinDataset&) = delete;

    // Returns {x, y} on `device`, mirrors get_batch(batch_size)
    std::pair<torch::Tensor, torch::Tensor> get_batch(int64_t batch_size, torch::Device device);

private:
    int fd_ = -1;
    const uint16_t* data_ = nullptr;
    size_t num_tokens_ = 0;
    int64_t block_size_;
};