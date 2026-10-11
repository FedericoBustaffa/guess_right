#include "parameter.hpp"

Parameter::Parameter(float data, bool requires_grad, float weight_decay)
    : Tensor(data, requires_grad), m_WeightDecay(weight_decay)
{
}

Parameter::Parameter(const Matrix& data, bool requires_grad, float weight_decay)
    : Tensor(data, requires_grad), m_WeightDecay(weight_decay)
{
}

Parameter::Parameter(Matrix&& data, bool requires_grad, float weight_decay)
    : Tensor(data, requires_grad), m_WeightDecay(weight_decay)
{
}
