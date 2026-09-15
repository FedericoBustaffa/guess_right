#include "optimizer.hpp"

// -----------------
// --- OPTIMIZER ---
// -----------------

Optimizer::Optimizer(const std::vector<Parameter>& parameters)
    : m_Parameters(parameters)
{
}

void Optimizer::zero_grad()
{
    for (Parameter& p : m_Parameters)
        p.zero_grad();
}

// -----------------------------------
// --- STOCHASTIC GRADIENT DESCENT ---
// -----------------------------------

StochasticGradientDescent::StochasticGradientDescent(
    const std::vector<Parameter>& parameters, float learning_rate)
    : Optimizer(parameters), m_LearningRate(learning_rate)
{
}

void StochasticGradientDescent::step()
{
    for (Parameter& p : m_Parameters)
        p.item() -= m_LearningRate * p.gradient();
}
