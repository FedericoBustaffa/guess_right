#include "modules.hpp"

#include <cmath>

std::vector<Tensor> Module::forward(const std::vector<Tensor>& x) const
{
    std::vector<Tensor> out;
    out.reserve(x.size());

    for (const Tensor& t : x)
        out.push_back(forward(t));

    return out;
}

// --------------
// --- LINEAR ---
// --------------
Linear::Linear(size_t in_features, size_t out_features)
    : w(NDArray::normal({out_features, in_features}, 0.0f,
                        std::sqrt(2.0f / in_features)),
        true, 1.0f),
      b(NDArray::zeros({out_features}), true, 0.0f)
{
}

std::vector<Parameter> Linear::parameters() const { return {w, b}; }

Tensor Linear::forward(const Tensor& x) const { return matmul(w, x) + b; }

// ----------------
// --- RESIDUAL ---
// ----------------
Tensor Residual::forward(const Tensor& x) const
{
    Tensor out = x;

    for (const auto& module : m_Modules)
        out = (*module)(out);

    return out + x;
}

std::vector<Parameter> Residual::parameters() const
{
    std::vector<Parameter> params;

    for (const auto& module : m_Modules)
    {
        auto p = module->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }

    return params;
}

// -------------
// --- MODEL ---
// -------------
Tensor Model::forward(const Tensor& x) const
{
    Tensor out = x;
    for (const auto& module : m_Modules)
        out = (*module)(out);

    return out;
}

std::vector<Parameter> Model::parameters() const
{
    std::vector<Parameter> params;

    for (const auto& module : m_Modules)
    {
        auto module_parameters = module->parameters();
        params.insert(params.end(), module_parameters.begin(),
                      module_parameters.end());
    }

    return params;
}
