#ifndef MODULES_HPP
#define MODULES_HPP

#include "parameter.hpp"
#include <vector>

// --------------
// --- MODULE ---
// --------------
class Module
{
public:
    Module() = default;

    virtual inline std::vector<Parameter> parameters() const { return {}; }

    virtual Tensor forward(const Tensor& x) const = 0;

    std::vector<Tensor> forward(const std::vector<Tensor>& x) const;

    inline Tensor operator()(const Tensor& x) const { return forward(x); }

    inline std::vector<Tensor> operator()(const std::vector<Tensor>& x) const
    {
        return forward(x);
    }

    virtual ~Module() = default;
};

// --------------
// --- LINEAR ---
// --------------
class Linear : public Module
{
public:
    Linear(size_t in_features, size_t out_features);

    inline std::vector<Parameter> parameters() const override;

    Tensor forward(const Tensor& x) const override;

    ~Linear() = default;

private:
    Parameter w;
    Parameter b;
};

// ----------------
// --- RESIDUAL ---
// ----------------
class Residual : public Module
{
public:
    Residual() = default;

    Tensor forward(const Tensor& x) const override;

    std::vector<Parameter> parameters() const override;

    template <typename T>
    void add(const T& module)
    {
        m_Modules.push_back(std::make_shared<T>(module));
    }

    template <typename T, typename... Args>
    void add(Args&&... args)
    {
        m_Modules.push_back(std::make_shared<T>(std::forward<Args>(args)...));
    }

private:
    std::vector<std::shared_ptr<Module>> m_Modules;
};

// ------------------
// --- SEQUENTIAL ---
// ------------------
class Sequential : public Module
{
public:
    Sequential() = default;

    Tensor forward(const Tensor& x) const override;

    std::vector<Parameter> parameters() const override;

    template <typename T>
    void add(const T& module)
    {
        m_Modules.push_back(std::make_shared<T>(module));
    }

    template <typename T, typename... Args>
    void add(Args&&... args)
    {
        m_Modules.push_back(std::make_shared<T>(std::forward<Args>(args)...));
    }

private:
    std::vector<std::shared_ptr<Module>> m_Modules;
};

#endif
