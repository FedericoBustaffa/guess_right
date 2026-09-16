#include "ndarray.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

size_t size_from_shape(const std::vector<size_t>& shape)
{
    size_t size = 1;
    for (const auto& dim : shape)
    {
        if (dim == 0)
            throw std::invalid_argument("dimension cannot be zero");
        size *= dim;
    }

    return size;
}

NDArray::NDArray(float data) : m_Shape({1, 1}), m_Size(1)
{
    m_Data = new float[1];
    *m_Data = data;
}

NDArray::NDArray(float* data, const std::vector<size_t>& shape)
    : m_Data(data), m_Size(size_from_shape(shape))
{
    if (data == nullptr)
        throw std::invalid_argument("null data pointer");

    if (std::find(shape.begin(), shape.end(), 0) != shape.end())
        throw std::invalid_argument("zero is not a valid dimension");

    if (shape.size() > 2)
        throw std::invalid_argument("multi-dimensional tensors not supported");

    if (shape.empty())
        m_Shape = {1, 1};
    else if (shape.size() == 1)
        m_Shape = {shape[0], 1};
    else
        m_Shape = shape;
}

NDArray::NDArray(const std::vector<float>& data,
                 const std::vector<size_t>& shape)
    : m_Data(nullptr), m_Size(size_from_shape(shape))
{
    m_Data = new float[m_Size];
    std::copy(data.begin(), data.end(), m_Data);

    if (shape.empty())
        m_Shape = {1, 1};
    else if (shape.size() == 1)
        m_Shape = {shape[0], 1};
    else
        m_Shape = shape;
}

NDArray::NDArray(const NDArray& other)
    : m_Shape(other.m_Shape), m_Size(other.m_Size)
{
    m_Data = new float[other.size()];
    std::copy(other.m_Data, other.m_Data + other.size(), m_Data);
}

NDArray::NDArray(NDArray&& other)
    : m_Data(other.m_Data), m_Shape(std::move(other.m_Shape)),
      m_Size(other.m_Size)
{
    other.m_Data = nullptr;
    other.m_Size = 0;
}

NDArray& NDArray::operator=(const NDArray& other)
{
    if (this == &other)
        return *this;

    delete[] m_Data;
    m_Data = new float[other.size()];
    std::copy(other.m_Data, other.m_Data + other.size(), m_Data);

    m_Shape = other.m_Shape;
    m_Size = other.m_Size;

    return *this;
}

NDArray& NDArray::operator=(NDArray&& other) noexcept
{
    if (this == &other)
        return *this;

    delete[] m_Data;

    m_Data = other.m_Data;
    m_Shape = std::move(other.m_Shape);
    m_Size = other.m_Size;

    other.m_Data = nullptr;
    other.m_Size = 0;

    return *this;
}

NDArray unary_operator(const NDArray& x, const std::function<float(float)>& op)
{
    float* data = new float[x.size()];
    for (size_t i = 0; i < x.size(); ++i)
        data[i] = op(x.m_Data[i]);

    return NDArray(data, x.m_Shape);
}

NDArray binary_operator(const NDArray& a, const NDArray& b,
                        const std::function<float(float, float)>& op)
{
    if (!a.is_scalar() && !b.is_scalar() && a.shape() != b.shape())
        throw std::invalid_argument(
            std::format("binary operator shape mismatch: {} and {}\n{}\n{}",
                        a.shape(), b.shape(), a, b));

    const size_t size = a.is_scalar() ? b.size() : a.size();
    float* data = new float[size];
    for (size_t i = 0; i < size; ++i)
    {
        const float x = a.is_scalar() ? a.m_Data[0] : a.m_Data[i];
        const float y = b.is_scalar() ? b.m_Data[0] : b.m_Data[i];
        data[i] = op(x, y);
    }

    return NDArray(data, a.is_scalar() ? b.shape() : a.shape());
}

NDArray NDArray::operator-() const
{

    return unary_operator(*this, [](float x) { return -x; });
}

NDArray operator+(const NDArray& a, const NDArray& b)
{
    return binary_operator(a, b, [](float a, float b) { return a + b; });
}

NDArray operator-(const NDArray& a, const NDArray& b)
{
    return binary_operator(a, b, [](float a, float b) { return a - b; });
}

NDArray operator*(const NDArray& a, const NDArray& b)
{
    return binary_operator(a, b, [](float a, float b) { return a * b; });
}

NDArray operator/(const NDArray& a, const NDArray& b)
{
    return binary_operator(a, b, [](float a, float b) { return a / b; });
}

NDArray& binary_assign_operator(NDArray& a, const NDArray& b,
                                const std::function<float(float, float)>& op)
{
    const bool b_singleton = b.size() == 1;

    if (!b_singleton && a.m_Shape != b.m_Shape)
        throw std::invalid_argument(std::format(
            "binary assign operator shape mismatch: {} and {}\n{}\n{}",
            a.shape(), b.shape(), a, b));

    if (b_singleton)
    {
        const float value = *b.m_Data;

        for (size_t i = 0; i < a.size(); ++i)
            a.m_Data[i] = op(a.m_Data[i], value);

        return a;
    }

    for (size_t i = 0; i < a.size(); ++i)
        a.m_Data[i] = op(a.m_Data[i], b.m_Data[i]);

    return a;
}

NDArray& NDArray::operator+=(const NDArray& other)
{
    return binary_assign_operator(*this, other,
                                  [](float a, float b) { return a + b; });
}

NDArray& NDArray::operator-=(const NDArray& other)
{
    return binary_assign_operator(*this, other,
                                  [](float a, float b) { return a - b; });
}

NDArray& NDArray::operator*=(const NDArray& other)
{
    return binary_assign_operator(*this, other,
                                  [](float a, float b) { return a * b; });
}

NDArray& NDArray::operator/=(const NDArray& other)
{
    return binary_assign_operator(*this, other,
                                  [](float a, float b) { return a / b; });
}

NDArray operator<(const NDArray& a, const NDArray& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a < b ? 1.0f : 0.0f; });
}

NDArray operator<=(const NDArray& a, const NDArray& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a <= b ? 1.0f : 0.0f; });
}

NDArray operator==(const NDArray& a, const NDArray& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a == b ? 1.0f : 0.0f; });
}

NDArray operator>=(const NDArray& a, const NDArray& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a >= b ? 1.0f : 0.0f; });
}

NDArray operator>(const NDArray& a, const NDArray& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a > b ? 1.0f : 0.0f; });
}

NDArray NDArray::transpose() const
{
    if (this->ndim() != 2)
        return *this;

    const size_t rows = m_Shape[0];
    const size_t cols = m_Shape[1];

    float* data = new float[m_Size];
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            data[j * rows + i] = m_Data[i * cols + j];

    return NDArray(data, {m_Shape[1], m_Shape[0]});
}

NDArray sum(const NDArray& a)
{
    float res = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        res += a.m_Data[i];

    return NDArray(res);
}

NDArray mean(const NDArray& a) { return sum(a) / a.size(); }

NDArray product(const NDArray& a)
{
    float res = 1.0f;
    for (size_t i = 0; i < a.size(); i++)
        res *= a.m_Data[i];

    return NDArray(res);
}

NDArray matmul(const NDArray& a, const NDArray& b)
{
    const size_t rows = a.shape()[0];
    const size_t inner = a.shape()[1];
    const size_t cols = b.shape()[1];

    if (inner != b.shape()[0])
        throw std::invalid_argument(std::format(
            "inner: incompatible shapes {} and {}", inner, b.shape()[0]));

    float* data = new float[rows * cols]();
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t k = 0; k < inner; ++k)
        {
            const float aik = a.m_Data[i * inner + k];
            for (size_t j = 0; j < cols; ++j)
                data[i * cols + j] += aik * b.m_Data[k * cols + j];
        }
    }
    return NDArray(data, {rows, cols});
}

NDArray pow(const NDArray& x, const NDArray& e)
{
    return binary_operator(x, e,
                           [](float a, float b) { return std::pow(a, b); });
}

NDArray abs(const NDArray& x)
{
    return unary_operator(x, [](float a) { return std::abs(a); });
}

NDArray sqrt(const NDArray& x)
{
    return unary_operator(x, [](float a) { return std::sqrt(a); });
}

NDArray log(const NDArray& x)
{
    return unary_operator(x, [](float a) { return std::log(a); });
}

NDArray exp(const NDArray& x)
{
    return unary_operator(x, [](float a) { return std::exp(a); });
}

NDArray sin(const NDArray& x)
{
    return unary_operator(x, [](float a) { return std::sin(a); });
}

NDArray cos(const NDArray& x)
{
    return unary_operator(x, [](float a) { return std::cos(a); });
}

NDArray tanh(const NDArray& x)
{
    return unary_operator(x, [](float a) { return std::tanh(a); });
}

NDArray maximum(const NDArray& a, const NDArray& b)
{
    return binary_operator(a, b,
                           [](float x, float y) { return x > y ? x : y; });
}

NDArray minimum(const NDArray& a, const NDArray& b)
{
    return binary_operator(a, b,
                           [](float x, float y) { return x < y ? x : y; });
}

NDArray round(const NDArray& a)
{
    return unary_operator(a, [](float x) { return std::round(x); });
}

NDArray::~NDArray() { delete[] m_Data; }

// --- STATIC ---

NDArray NDArray::zeros(const std::vector<size_t>& shape)
{
    size_t size = size_from_shape(shape);
    float* data = new float[size];
    std::fill(data, data + size, 0.0f);

    return NDArray(data, shape);
}

NDArray NDArray::zeros_like(const NDArray& t)
{
    float* data = new float[t.size()];
    std::fill(data, data + t.size(), 0.0f);

    return NDArray(data, t.shape());
}

NDArray NDArray::ones(const std::vector<size_t>& shape)
{
    size_t size = size_from_shape(shape);
    float* data = new float[size];
    std::fill(data, data + size, 1.0f);

    return NDArray(data, shape);
}

NDArray NDArray::ones_like(const NDArray& t)
{
    float* data = new float[t.size()];
    std::fill(data, data + t.size(), 1.0f);

    return NDArray(data, t.shape());
}

NDArray NDArray::eye(size_t dim)
{
    if (dim == 0)
        throw std::invalid_argument("dim cannot be zero");

    float* data = new float[dim * dim];
    std::fill(data, data + dim * dim, 0.0f);
    for (size_t i = 0; i < dim; i++)
        data[i * dim + i] = 1.0f;

    return NDArray(data, {dim, dim});
}

NDArray NDArray::linspace(float start, float end, size_t samples)
{
    if (samples == 0)
        throw std::invalid_argument("samples cannot be zero");

    float* data = new float[samples];

    if (samples == 1)
    {
        data[0] = start;
        return NDArray(data, {1});
    }

    const float step = (end - start) / static_cast<float>(samples - 1);

    for (size_t i = 0; i < samples; ++i)
        data[i] = start + static_cast<float>(i) * step;

    return NDArray(data, {samples});
}

void NDArray::seed(int s) { rng.seed(s); }

NDArray NDArray::normal(const std::vector<size_t>& shape, float mu, float sigma)
{
    size_t size = 1;
    for (const size_t& dim : shape)
    {
        if (dim == 0)
            throw std::invalid_argument("dimension cannot be zero");
        size *= dim;
    }

    std::normal_distribution<float> distribution(mu, sigma);
    float* data = new float[size];
    for (size_t i = 0; i < size; i++)
        data[i] = distribution(rng);

    return NDArray(data, shape);
}

NDArray NDArray::uniform(const std::vector<size_t>& shape, float a, float b)
{
    size_t size = 1;
    for (const size_t& dim : shape)
    {
        if (dim == 0)
            throw std::invalid_argument("dimension cannot be zero");
        size *= dim;
    }

    std::uniform_real_distribution<float> distribution(a, b);
    float* data = new float[size];
    for (size_t i = 0; i < size; i++)
        data[i] = distribution(rng);

    return NDArray(data, shape);
}

std::default_random_engine NDArray::rng((std::random_device())());
