#ifndef PARAMETER_HPP
#define PARAMETER_HPP

#include "tensor.hpp"

class Parameter : public Tensor
{
public:
    Parameter(float data, bool requires_grad = true, float weight_decay = 1.0f);

    Parameter(const NDArray& data, bool requires_grad = true,
              float weight_decay = 1.0f);

    Parameter(NDArray&& data, bool requires_grad = true,
              float weight_decay = 1.0f);

    inline float weight_decay() const { return m_WeightDecay; }

    ~Parameter() = default;

private:
    float m_WeightDecay;
};

#endif
