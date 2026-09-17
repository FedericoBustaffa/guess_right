#include "data.hpp"

#include <algorithm>
#include <stdexcept>

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

// --------------
// --- LOADER ---
// --------------
Loader::Loader(const Dataset& dataset, size_t batch_size, bool shuffle)
    : m_Dataset(dataset), m_BatchSize(batch_size), m_Shuffle(shuffle)
{
    if (batch_size == 0)
        throw std::invalid_argument("batch size zero not allowed");

    m_Indices.reserve(dataset.size());
    for (size_t i = 0; i < dataset.size(); i++)
        m_Indices.emplace_back(i);
}

Dataset Loader::get()
{
    std::vector<Tensor> x;
    x.reserve(m_BatchSize);
    std::vector<Tensor> y;
    y.reserve(m_BatchSize);

    const size_t batch_end =
        std::min(m_Current + m_BatchSize, m_Dataset.size());

    for (; m_Current < batch_end; m_Current++)
    {
        x.push_back(m_Dataset.inputs()[m_Current]);
        y.push_back(m_Dataset.targets()[m_Current]);
    }

    if (m_Current >= batch_end)
        m_Current = 0;

    return Dataset(std::move(x), std::move(y));
}
