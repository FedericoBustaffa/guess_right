#ifndef DATA_HPP
#define DATA_HPP

#include <vector>

#include "tensor.hpp"

class Dataset
{
public:
    Dataset(const std::vector<Tensor>& x, const std::vector<Tensor>& y);

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
    Loader(const Dataset& dataset, size_t batch_size, bool shuffle);

    ~Loader() = default;

private:
    std::vector<std::vector<Tensor>> m_Batches;
};

#endif
