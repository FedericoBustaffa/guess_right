#include "data.hpp"

// ---------------
// --- DATASET ---
// ---------------
Dataset::Dataset(const std::vector<Tensor>& x, const std::vector<Tensor>& y)
    : m_Inputs(x), m_Targets(y)
{
}

Dataset::Dataset(std::vector<Tensor>&& x, std::vector<Tensor>&& y)
    : m_Inputs(std::move(x)), m_Targets(std::move(y))
{
}

// ---------------
// --- DATASET ---
// ---------------
Loader::Loader(const Dataset& dataset, size_t batch_size, bool shuffle)
    : m_Dataset(dataset), m_BatchSize(batch_size), m_Shuffle(shuffle)
{
}
