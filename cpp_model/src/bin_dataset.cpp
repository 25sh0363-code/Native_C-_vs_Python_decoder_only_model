#include "bin_dataset.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>

BinDataset::BinDataset(const std::string& path, int64_t block_size)
    : block_size_(block_size) {

    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("Failed to open " + path);
    }

    struct stat st;
    if (fstat(fd_, &st) < 0) {
        close(fd_);
        throw std::runtime_error("fstat failed for " + path);
    }
    size_t file_size = static_cast<size_t>(st.st_size);
    num_tokens_ = file_size / sizeof(uint16_t);

    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed for " + path);
    }
    data_ = reinterpret_cast<const uint16_t*>(mapped);
}

BinDataset::~BinDataset() {
    if (data_) munmap((void*)data_, num_tokens_ * sizeof(uint16_t));
    if (fd_ >= 0) close(fd_);
}

std::pair<torch::Tensor, torch::Tensor> BinDataset::get_batch(int64_t batch_size, torch::Device device) {
    int64_t max_start = static_cast<int64_t>(num_tokens_) - block_size_ - 1;
    if (max_start < 0) {
        throw std::runtime_error("BinDataset: file has fewer tokens than block_size");
    }

    auto starts = torch::randint(0, max_start + 1, {batch_size}, torch::kLong);

    auto x = torch::empty({batch_size, block_size_}, torch::kLong);
    auto y = torch::empty({batch_size, block_size_}, torch::kLong);

    // Raw pointer writes rather than accessor<> — this runs every training
    // step, so per-element bounds-checked indexing overhead adds up.
    int64_t* x_ptr = x.data_ptr<int64_t>();
    int64_t* y_ptr = y.data_ptr<int64_t>();
    const int64_t* starts_ptr = starts.data_ptr<int64_t>();

    for (int64_t b = 0; b < batch_size; ++b) {
        int64_t s = starts_ptr[b];
        int64_t* x_row = x_ptr + b * block_size_;
        int64_t* y_row = y_ptr + b * block_size_;
        for (int64_t t = 0; t < block_size_; ++t) {
            x_row[t] = static_cast<int64_t>(data_[s + t]);
            y_row[t] = static_cast<int64_t>(data_[s + t + 1]);
        }
    }

    return {x.to(device), y.to(device)};
}