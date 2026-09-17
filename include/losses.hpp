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

    virtual Tensor forward(const std::vector<Tensor>& predicted,
                           const std::vector<Tensor>& target) = 0;

    Tensor operator()(const Tensor& predicted, const Tensor& target)
    {
        return forward(predicted, target);
    }

    Tensor operator()(const std::vector<Tensor>& predicted,
                      const std::vector<Tensor>& target)
    {
        return forward(predicted, target);
    }

    virtual ~Loss() = default;
};

// --------------------------
// --- MEAN SQUARED ERROR ---
// --------------------------
class MeanSquaredError : public Loss
{
public:
    MeanSquaredError() = default;

    Tensor forward(const Tensor& predicted, const Tensor& target)
    {
        return mean(pow(predicted - target, 2.0f));
    }

    Tensor forward(const std::vector<Tensor>& predicted,
                   const std::vector<Tensor>& target)
    {
        Tensor loss = 0.0f;
        const size_t n = predicted.size();
        for (size_t i = 0; i < n; i++)
            loss += mean(pow(predicted[i] - target[i], 2.0f)) / n;

        return loss;
    }

    ~MeanSquaredError() = default;
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

    Tensor forward(const std::vector<Tensor>& predicted,
                   const std::vector<Tensor>& target)
    {
        Tensor loss = 0.0f;
        const size_t n = predicted.size();
        for (size_t i = 0; i < n; i++)
            loss += mean(forward(predicted, target)) / n;

        return loss;
    }

    ~BinaryCrossEntropy() = default;
};

// -------------------------------
// --- NEGATIVE LOG-LIKELIHOOD ---
// -------------------------------
class CrossEntropy : public Loss
{
public:
    CrossEntropy() = default;

    Tensor forward(const Tensor& predicted, const Tensor& target)
    {
        return -sum(target * log(predicted));
    }

    Tensor forward(const std::vector<Tensor>& predicted,
                   const std::vector<Tensor>& target)
    {
        Tensor loss = 0.0f;
        const size_t n = predicted.size();
        for (size_t i = 0; i < n; i++)
            loss += mean(forward(predicted, target)) / n;

        return loss;
    }

    ~CrossEntropy() = default;
};

#endif
