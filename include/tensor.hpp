#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <functional>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ndarray.hpp"
#include "node.hpp"

class Tensor
{
public:
    Tensor(float data, bool requires_grad = false);
    Tensor(const NDArray& data, bool requires_grad = false);
    Tensor(NDArray&& data, bool requires_grad = false);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    inline const NDArray& item() const { return m_Node->data; }
    inline NDArray& item() { return m_Node->data; }
    inline const NDArray& gradient() const { return m_Node->grad; }
    inline NDArray& gradient() { return m_Node->grad; }
    inline bool requires_grad() const { return m_Node->requires_grad; }
    inline const std::vector<size_t>& shape() const
    {
        return m_Node->data.shape();
    }
    inline size_t ndim() const { return m_Node->data.ndim(); }
    inline size_t size() const { return m_Node->data.size(); }
    inline bool is_scalar() const { return m_Node->data.is_scalar(); }

    void backward();
    void zero_grad() { m_Node->grad *= 0.0f; }

    //  Operators
    friend Tensor operator-(const Tensor& a);
    friend Tensor operator+(const Tensor& a, const Tensor& b);
    friend Tensor operator-(const Tensor& a, const Tensor& b);
    friend Tensor operator*(const Tensor& a, const Tensor& b);
    friend Tensor operator/(const Tensor& a, const Tensor& b);

    // Assign operators
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    // Functions
    Tensor transpose() const;
    friend Tensor sum(const Tensor& x);
    friend Tensor mean(const Tensor& x);
    friend Tensor product(const Tensor& x);
    friend Tensor matmul(const Tensor& a, const Tensor& b);
    friend Tensor pow(const Tensor& x, const Tensor& e);
    friend Tensor log(const Tensor& x);
    friend Tensor exp(const Tensor& x);
    friend Tensor sin(const Tensor& x);
    friend Tensor cos(const Tensor& x);
    friend Tensor tanh(const Tensor& x);
    friend Tensor maximum(const Tensor& a, const Tensor& b);
    friend Tensor minimum(const Tensor& a, const Tensor& b);

    // not part of computational graph
    friend Tensor round(const Tensor& a) { return round(a.item()); }

    virtual ~Tensor() = default;

private:
    Tensor(const std::shared_ptr<Node>& node);

    friend Tensor unary_operator(
        const Tensor& a, Operation operation,
        const std::function<NDArray(const NDArray&)>& func);

    friend Tensor binary_operator(
        const Tensor& a, const Tensor& b, Operation operation,
        const std::function<NDArray(const NDArray&, const NDArray&)>& func);

    friend Tensor& binary_assign_operator(
        Tensor& a, const Tensor& b, Operation operation,
        const std::function<NDArray(const NDArray&, const NDArray&)>& func);

private: // Members
    std::shared_ptr<Node> m_Node;

public: // STATIC
    static Tensor zeros(const std::vector<size_t>& shape = {},
                        bool requires_grad = false);
    static Tensor zeros_like(const Tensor& t, bool requires_grad = false);

    static Tensor ones(const std::vector<size_t>& shape = {},
                       bool requires_grad = false);
    static Tensor ones_like(const Tensor& t, bool requires_grad = false);

    static Tensor eye(size_t dim, bool requires_grad = false);
    static Tensor linspace(float start, float end, size_t samples = 50,
                           bool requires_grad = false);

    static void seed(int s);

    static Tensor normal(const std::vector<size_t>& shape = {}, float mu = 0.0f,
                         float sigma = 1.0f, bool requires_grad = false);

    static Tensor uniform(const std::vector<size_t>& shape = {}, float a = 0.0f,
                          float b = 1.0f, bool requires_grad = false);
};

#endif
