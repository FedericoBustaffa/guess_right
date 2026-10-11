#include "matrix.hpp"

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

Matrix::Matrix(float data) : m_Shape({1, 1}), m_Size(1)
{
    m_Data = new float[1];
    *m_Data = data;
}

Matrix::Matrix(float* data, const std::vector<size_t>& shape)
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

Matrix::Matrix(const std::vector<float>& data, const std::vector<size_t>& shape)
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

Matrix::Matrix(const Matrix& other)
    : m_Shape(other.m_Shape), m_Size(other.m_Size)
{
    m_Data = new float[other.size()];
    std::copy(other.m_Data, other.m_Data + other.size(), m_Data);
}

Matrix::Matrix(Matrix&& other)
    : m_Data(other.m_Data), m_Shape(std::move(other.m_Shape)),
      m_Size(other.m_Size)
{
    other.m_Data = nullptr;
    other.m_Size = 0;
}

Matrix& Matrix::operator=(const Matrix& other)
{
    if (this == &other)
        return *this;

    if (m_Size != other.size())
    {
        delete[] m_Data;
        m_Data = new float[other.size()];
    }
    std::copy(other.m_Data, other.m_Data + other.size(), m_Data);

    m_Shape = other.m_Shape;
    m_Size = other.m_Size;

    return *this;
}

Matrix& Matrix::operator=(Matrix&& other) noexcept
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

Matrix::operator float() const
{
    if (!is_scalar())
        throw std::logic_error("Matrix: cannot convert non-scalar to float");

    return m_Data[0];
}

Matrix Matrix::operator-() const
{

    return unary_operator(*this, [](float x) { return -x; });
}

Matrix operator+(const Matrix& a, const Matrix& b)
{
    return binary_operator(a, b, [](float a, float b) { return a + b; });
}

Matrix operator-(const Matrix& a, const Matrix& b)
{
    return binary_operator(a, b, [](float a, float b) { return a - b; });
}

Matrix operator*(const Matrix& a, const Matrix& b)
{
    return binary_operator(a, b, [](float a, float b) { return a * b; });
}

Matrix operator/(const Matrix& a, const Matrix& b)
{
    return binary_operator(a, b, [](float a, float b) { return a / b; });
}

Matrix& Matrix::operator+=(const Matrix& other)
{
    return binary_assign_operator(*this, other,
                                  [](float a, float b) { return a + b; });
}

Matrix& Matrix::operator-=(const Matrix& other)
{
    return binary_assign_operator(*this, other,
                                  [](float a, float b) { return a - b; });
}

Matrix& Matrix::operator*=(const Matrix& other)
{
    return binary_assign_operator(*this, other,
                                  [](float a, float b) { return a * b; });
}

Matrix& Matrix::operator/=(const Matrix& other)
{
    return binary_assign_operator(*this, other,
                                  [](float a, float b) { return a / b; });
}

Matrix operator<(const Matrix& a, const Matrix& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a < b ? 1.0f : 0.0f; });
}

Matrix operator<=(const Matrix& a, const Matrix& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a <= b ? 1.0f : 0.0f; });
}

Matrix operator==(const Matrix& a, const Matrix& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a == b ? 1.0f : 0.0f; });
}

Matrix operator>=(const Matrix& a, const Matrix& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a >= b ? 1.0f : 0.0f; });
}

Matrix operator>(const Matrix& a, const Matrix& b)
{
    return binary_operator(
        a, b, [](float a, float b) { return a > b ? 1.0f : 0.0f; });
}

Matrix Matrix::transpose() const
{
    if (this->ndim() != 2)
        return *this;

    const size_t rows = m_Shape[0];
    const size_t cols = m_Shape[1];

    float* data = new float[m_Size];
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            data[j * rows + i] = m_Data[i * cols + j];

    return Matrix(data, {m_Shape[1], m_Shape[0]});
}

Matrix sum(const Matrix& a)
{
    float res = 0.0f;
    for (size_t i = 0; i < a.size(); i++)
        res += a.m_Data[i];

    return Matrix(res);
}

Matrix mean(const Matrix& a) { return sum(a) / a.size(); }

Matrix product(const Matrix& a)
{
    float res = 1.0f;
    for (size_t i = 0; i < a.size(); i++)
        res *= a.m_Data[i];

    return Matrix(res);
}

Matrix matmul(const Matrix& a, const Matrix& b)
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

    return Matrix(data, {rows, cols});
}

Matrix pow(const Matrix& x, const Matrix& e)
{
    return binary_operator(x, e,
                           [](float a, float b) { return std::pow(a, b); });
}

Matrix abs(const Matrix& x)
{
    return unary_operator(x, [](float a) { return std::abs(a); });
}

Matrix sqrt(const Matrix& x)
{
    return unary_operator(x, [](float a) { return std::sqrt(a); });
}

Matrix log(const Matrix& x)
{
    return unary_operator(x, [](float a) { return std::log(a); });
}

Matrix exp(const Matrix& x)
{
    return unary_operator(x, [](float a) { return std::exp(a); });
}

Matrix sin(const Matrix& x)
{
    return unary_operator(x, [](float a) { return std::sin(a); });
}

Matrix cos(const Matrix& x)
{
    return unary_operator(x, [](float a) { return std::cos(a); });
}

Matrix tanh(const Matrix& x)
{
    return unary_operator(x, [](float a) { return std::tanh(a); });
}

Matrix maximum(const Matrix& a, const Matrix& b)
{
    return binary_operator(a, b,
                           [](float x, float y) { return x > y ? x : y; });
}

Matrix minimum(const Matrix& a, const Matrix& b)
{
    return binary_operator(a, b,
                           [](float x, float y) { return x < y ? x : y; });
}

Matrix round(const Matrix& a)
{
    return unary_operator(a, [](float x) { return std::round(x); });
}

Matrix::~Matrix() { delete[] m_Data; }

// --- STATIC ---

Matrix Matrix::zeros(const std::vector<size_t>& shape)
{
    size_t size = size_from_shape(shape);
    float* data = new float[size];
    std::fill(data, data + size, 0.0f);

    return Matrix(data, shape);
}

Matrix Matrix::zeros_like(const Matrix& t)
{
    float* data = new float[t.size()];
    std::fill(data, data + t.size(), 0.0f);

    return Matrix(data, t.shape());
}

Matrix Matrix::ones(const std::vector<size_t>& shape)
{
    size_t size = size_from_shape(shape);
    float* data = new float[size];
    std::fill(data, data + size, 1.0f);

    return Matrix(data, shape);
}

Matrix Matrix::ones_like(const Matrix& t)
{
    float* data = new float[t.size()];
    std::fill(data, data + t.size(), 1.0f);

    return Matrix(data, t.shape());
}

Matrix Matrix::eye(size_t dim)
{
    if (dim == 0)
        throw std::invalid_argument("dim cannot be zero");

    float* data = new float[dim * dim];
    std::fill(data, data + dim * dim, 0.0f);
    for (size_t i = 0; i < dim; i++)
        data[i * dim + i] = 1.0f;

    return Matrix(data, {dim, dim});
}

Matrix Matrix::linspace(float start, float end, size_t samples)
{
    if (samples == 0)
        throw std::invalid_argument("samples cannot be zero");

    float* data = new float[samples];

    if (samples == 1)
    {
        data[0] = start;
        return Matrix(data, {1});
    }

    const float step = (end - start) / static_cast<float>(samples - 1);

    for (size_t i = 0; i < samples; ++i)
        data[i] = start + static_cast<float>(i) * step;

    return Matrix(data, {samples});
}

void Matrix::seed(int s) { rng.seed(s); }

Matrix Matrix::normal(const std::vector<size_t>& shape, float mu, float sigma)
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

    return Matrix(data, shape);
}

Matrix Matrix::uniform(const std::vector<size_t>& shape, float a, float b)
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

    return Matrix(data, shape);
}

std::default_random_engine Matrix::rng((std::random_device())());
