#include "tensor.hpp"

#include "computational_graph.hpp"
#include "node.hpp"

Tensor::Tensor(const NDArray& data, bool requires_grad)
{
    m_Node = std::make_shared<Node>(data, requires_grad, Operation::Nop,
                                    std::vector<std::shared_ptr<Node>>{});
}

Tensor::Tensor(const std::shared_ptr<Node>& node) : m_Node(node) {}

Tensor::Tensor(float data, bool requires_grad)
{
    m_Node = std::make_shared<Node>(data, requires_grad, Operation::Nop,
                                    std::vector<std::shared_ptr<Node>>{});
}

Tensor::Tensor(NDArray&& data, bool requires_grad)
{
    m_Node =
        std::make_shared<Node>(std::move(data), requires_grad, Operation::Nop,
                               std::vector<std::shared_ptr<Node>>{});
}

Tensor::Tensor(const Tensor& other) : m_Node(other.m_Node) {}

Tensor::Tensor(Tensor&& other) noexcept : m_Node(std::move(other.m_Node)) {}

void Tensor::backward() { ComputationalGraph::backward(m_Node); }

Tensor unary_operator(const Tensor& a, Operation operation,
                      const std::function<NDArray(const NDArray&)>& func)
{
    NDArray data = func(a.item());

    if (ComputationalGraph::no_grad())
        return Tensor(std::move(data), false);

    std::shared_ptr<Node> node =
        std::make_shared<Node>(std::move(data), a.requires_grad(), operation,
                               std::vector<std::shared_ptr<Node>>{a.m_Node});

    return Tensor(node);
}

Tensor binary_operator(
    const Tensor& a, const Tensor& b, Operation operation,
    const std::function<NDArray(const NDArray&, const NDArray&)>& func)
{
    NDArray data = func(a.item(), b.item());

    if (ComputationalGraph::no_grad())
        return Tensor(std::move(data), false);

    bool requires_grad = a.requires_grad() || b.requires_grad();
    std::shared_ptr<Node> node = std::make_shared<Node>(
        std::move(data), requires_grad, operation,
        std::vector<std::shared_ptr<Node>>{a.m_Node, b.m_Node});

    return Tensor(node);
}

Tensor& binary_assign_operator(
    Tensor& a, const Tensor& b, Operation operation,
    const std::function<NDArray(const NDArray&, const NDArray&)>& func)
{
    NDArray data = func(a.item(), b.item());

    if (ComputationalGraph::no_grad())
    {
        a.m_Node->data = std::move(data);
        return a;
    }

    std::shared_ptr<Node> old = a.m_Node;
    bool requires_grad = a.requires_grad() || b.requires_grad();

    a.m_Node = std::make_shared<Node>(
        std::move(data), requires_grad, operation,
        std::vector<std::shared_ptr<Node>>{old, b.m_Node});

    return a;
}

Tensor operator-(const Tensor& a)
{
    return unary_operator(a, Operation::Neg,
                          [](const NDArray& a) { return -a; });
}

Tensor operator+(const Tensor& a, const Tensor& b)
{
    return binary_operator(
        a, b, Operation::Add,
        [](const NDArray& a, const NDArray& b) { return a + b; });
}

Tensor operator-(const Tensor& a, const Tensor& b)
{
    return binary_operator(
        a, b, Operation::Sub,
        [](const NDArray& a, const NDArray& b) { return a - b; });
}

Tensor operator*(const Tensor& a, const Tensor& b)
{
    return binary_operator(
        a, b, Operation::Mul,
        [](const NDArray& a, const NDArray& b) { return a * b; });
}

Tensor operator/(const Tensor& a, const Tensor& b)
{
    return binary_operator(
        a, b, Operation::Div,
        [](const NDArray& a, const NDArray& b) { return a / b; });
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this == &other)
        return *this;

    m_Node = other.m_Node;

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this == &other)
        return *this;

    m_Node = std::move(other.m_Node);

    return *this;
}

Tensor& Tensor::operator+=(const Tensor& other)
{
    return binary_assign_operator(
        *this, other, Operation::Add,
        [](const NDArray& a, const NDArray& b) { return a + b; });
}

Tensor& Tensor::operator-=(const Tensor& other)
{
    return binary_assign_operator(
        *this, other, Operation::Sub,
        [](const NDArray& a, const NDArray& b) { return a - b; });
}

Tensor& Tensor::operator*=(const Tensor& other)
{
    return binary_assign_operator(
        *this, other, Operation::Mul,
        [](const NDArray& a, const NDArray& b) { return a * b; });
}

Tensor& Tensor::operator/=(const Tensor& other)
{
    return binary_assign_operator(
        *this, other, Operation::Div,
        [](const NDArray& a, const NDArray& b) { return a / b; });
}

Tensor Tensor::transpose() const
{
    return unary_operator(*this, Operation::Transpose,
                          [](const NDArray& x) { return x.transpose(); });
}

Tensor sum(const Tensor& x)
{
    return unary_operator(x, Operation::Sum,
                          [](const NDArray& x) { return sum(x); });
}

Tensor mean(const Tensor& x)
{
    return unary_operator(x, Operation::Mean,
                          [](const NDArray& x) { return mean(x); });
}

Tensor product(const Tensor& x)
{
    return unary_operator(x, Operation::Product,
                          [](const NDArray& x) { return product(x); });
}

Tensor matmul(const Tensor& a, const Tensor& b)
{
    return binary_operator(
        a, b, Operation::Matmul,
        [](const NDArray& a, const NDArray& b) { return matmul(a, b); });
}

Tensor pow(const Tensor& x, const Tensor& e)
{
    return binary_operator(
        x, e, Operation::Pow,
        [](const NDArray& x, const NDArray& e) { return pow(x, e); });
}

Tensor log(const Tensor& x)
{
    return unary_operator(x, Operation::Log,
                          [](const NDArray& x) { return log(x); });
}

Tensor exp(const Tensor& x)
{
    return unary_operator(x, Operation::Exp,
                          [](const NDArray& x) { return exp(x); });
}

Tensor sin(const Tensor& x)
{
    return unary_operator(x, Operation::Sin,
                          [](const NDArray& x) { return sin(x); });
}

Tensor cos(const Tensor& x)
{
    return unary_operator(x, Operation::Cos,
                          [](const NDArray& x) { return cos(x); });
}

Tensor tanh(const Tensor& x)
{
    return unary_operator(x, Operation::Tanh,
                          [](const NDArray& x) { return tanh(x); });
}

Tensor maximum(const Tensor& a, const Tensor& b)
{
    return binary_operator(
        a, b, Operation::Maximum,
        [](const NDArray& a, const NDArray& b) { return maximum(a, b); });
}

Tensor minimum(const Tensor& a, const Tensor& b)
{
    return binary_operator(
        a, b, Operation::Minimum,
        [](const NDArray& a, const NDArray& b) { return minimum(a, b); });
}

Tensor Tensor::zeros(const std::vector<size_t>& shape, bool requires_grad)
{
    return Tensor(NDArray::zeros(shape), requires_grad);
}

Tensor Tensor::zeros_like(const Tensor& t, bool requires_grad)
{
    return Tensor(NDArray::zeros_like(t.item()), requires_grad);
}

Tensor Tensor::ones(const std::vector<size_t>& shape, bool requires_grad)
{
    return Tensor(NDArray::ones(shape), requires_grad);
}

Tensor Tensor::ones_like(const Tensor& t, bool requires_grad)
{
    return Tensor(NDArray::ones_like(t.item()), requires_grad);
}

Tensor Tensor::eye(size_t dim, bool requires_grad)
{
    return Tensor(NDArray::eye(dim), requires_grad);
}

Tensor Tensor::linspace(float start, float end, size_t samples,
                        bool requires_grad)
{
    return Tensor(NDArray::linspace(start, end, samples), requires_grad);
}

void Tensor::seed(int s) { NDArray::seed(s); }

Tensor Tensor::normal(const std::vector<size_t>& shape, float mu, float sigma,
                      bool requires_grad)
{
    return Tensor(NDArray::normal(shape, mu, sigma), requires_grad);
}

Tensor Tensor::uniform(const std::vector<size_t>& shape, float a, float b,
                       bool requires_grad)
{
    return Tensor(NDArray::uniform(shape, a, b), requires_grad);
}
