#ifndef LOSSES_HPP
#define LOSSES_HPP

#include "tensor.hpp"

// ------------
// --- LOSS ---
// ------------
class Loss
{
public:
    Loss() = default;

    virtual Tensor forward(const Tensor& predicted, const Tensor& target) = 0;

    Tensor operator()(const Tensor& predicted, const Tensor& target)
    {
        return forward(predicted, target);
    }

    virtual ~Loss() = default;
};

// ---------------------
// --- SQUARED ERROR ---
// ---------------------
class SquaredError : public Loss
{
public:
    SquaredError() = default;

    Tensor forward(const Tensor& predicted, const Tensor& target)
    {
        return sum(pow(predicted - target, 2.0f));
    }

    ~SquaredError() = default;
};

// ----------------------------
// --- BINARY CROSS ENTROPY ---
// ----------------------------
class BinaryCrossEntropy : public Loss
{
public:
    BinaryCrossEntropy() = default;

    Tensor forward(const Tensor& predicted, const Tensor& target)
    {
        return -sum(target * log(predicted) +
                    (1 - target) * log(1 - predicted));
    }

    ~BinaryCrossEntropy() = default;
};

// ---------------------
// --- CROSS ENTROPY ---
// ---------------------
class CrossEntropy : public Loss
{
public:
    CrossEntropy() = default;

    Tensor forward(const Tensor& predicted, const Tensor& target)
    {
        return -sum(target * log(predicted));
    }

    ~CrossEntropy() = default;
};

#endif
