#ifndef DATA_HPP
#define DATA_HPP

#include <random>
#include <vector>

#include "tensor.hpp"

class Dataset
{
public:
    Dataset(const std::vector<Tensor>& x, const std::vector<Tensor>& y);

    Dataset(std::vector<Tensor>&& x, std::vector<Tensor>&& y);

    inline size_t size() const { return m_Inputs.size(); }

    inline const std::vector<Tensor>& inputs() const { return m_Inputs; }

    inline const std::vector<Tensor>& targets() const { return m_Targets; }

    ~Dataset() = default;

private:
    std::vector<Tensor> m_Inputs;
    std::vector<Tensor> m_Targets;
};

class Loader
{
public:
    Loader(const Dataset& dataset, size_t batch_size = 1, bool shuffle = false);

    ~Loader() = default;

private:
    const Dataset& m_Dataset;
    size_t m_BatchSize = 1;
    bool m_Shuffle = false;

    std::vector<size_t> m_Indices;
    std::mt19937 m_Rng;
};

#endif
