#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "modules.hpp"

// ---------------
// --- SIGMOID ---
// ---------------
class Sigmoid : public Module
{
public:
    Sigmoid() = default;

    Tensor forward(const Tensor& x) const override
    {
        return 1.0f / (1.0f + exp(-x));
    }
};

// ---------------
// --- TANH ---
// ---------------
class Tanh : public Module
{
public:
    Tanh() = default;

    Tensor forward(const Tensor& x) const override { return tanh(x); }
};

// ---------------
// --- RELU ---
// ---------------
class ReLU : public Module
{
public:
    ReLU() = default;

    Tensor forward(const Tensor& x) const override { return maximum(0, x); }
};

// ---------------
// --- SOFTMAX ---
// ---------------
class Softmax : public Module
{
public:
    Softmax() = default;

    Tensor forward(const Tensor& x) const override
    {
        Tensor e = exp(x);
        return e / sum(e);
    }
};

#endif
