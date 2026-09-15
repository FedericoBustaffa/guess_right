#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "parameter.hpp"

class Optimizer
{
public:
    Optimizer(const std::vector<Parameter>& parameters);

    inline const std::vector<Parameter>& parameters() const
    {
        return m_Parameters;
    }

    void zero_grad();

    virtual void step() = 0;

    virtual ~Optimizer() = default;

protected:
    std::vector<Parameter> m_Parameters;
};

class StochasticGradientDescent : public Optimizer
{
public:
    StochasticGradientDescent(const std::vector<Parameter>& parameters,
                              float learning_rate = 1e-3);

    void step() override;

    ~StochasticGradientDescent() = default;

private:
    float m_LearningRate;
};

#endif
