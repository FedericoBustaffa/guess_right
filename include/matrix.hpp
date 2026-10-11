#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <format>
#include <random>
#include <vector>

class Matrix
{
public:
    Matrix(float data);
    Matrix(float* data, const std::vector<size_t>& shape);
    Matrix(const std::vector<float>& data, const std::vector<size_t>& shape);
    Matrix(const Matrix& other);
    Matrix(Matrix&& other);

    inline const float* data() const { return m_Data; }
    inline const std::vector<size_t>& shape() const { return m_Shape; }
    inline size_t ndim() const { return m_Shape.size(); }
    inline size_t size() const { return m_Size; }
    inline bool is_scalar() const { return m_Shape[0] == 1 && m_Shape[1] == 1; }

    explicit operator float() const;

    //  Operators
    Matrix operator-() const;
    friend Matrix operator+(const Matrix& a, const Matrix& b);
    friend Matrix operator-(const Matrix& a, const Matrix& b);
    friend Matrix operator*(const Matrix& a, const Matrix& b);
    friend Matrix operator/(const Matrix& a, const Matrix& b);

    // Assign operators
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other) noexcept;
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(const Matrix& other);
    Matrix& operator/=(const Matrix& other);

    // Logic operators
    friend Matrix operator<(const Matrix& a, const Matrix& b);
    friend Matrix operator<=(const Matrix& a, const Matrix& b);
    friend Matrix operator==(const Matrix& a, const Matrix& b);
    friend Matrix operator>=(const Matrix& a, const Matrix& b);
    friend Matrix operator>(const Matrix& a, const Matrix& b);

    // functions
    Matrix transpose() const;
    friend Matrix sum(const Matrix& a);
    friend Matrix mean(const Matrix& a);
    friend Matrix product(const Matrix& a);
    friend Matrix matmul(const Matrix& a, const Matrix& b);
    friend Matrix pow(const Matrix& x, const Matrix& e);
    friend Matrix abs(const Matrix& x);
    friend Matrix sqrt(const Matrix& x);
    friend Matrix log(const Matrix& x);
    friend Matrix exp(const Matrix& x);
    friend Matrix sin(const Matrix& x);
    friend Matrix cos(const Matrix& x);
    friend Matrix tanh(const Matrix& x);
    friend Matrix maximum(const Matrix& a, const Matrix& b);
    friend Matrix minimum(const Matrix& a, const Matrix& b);

    friend Matrix round(const Matrix& a);

    ~Matrix();

private:
    template <typename Op>
    friend Matrix unary_operator(const Matrix& x, Op op);

    template <typename Op>
    friend Matrix binary_operator(const Matrix& a, const Matrix& b, Op op);

    template <typename Op>
    friend Matrix& binary_assign_operator(Matrix& a, const Matrix& b, Op op);

private:
    float* m_Data;
    std::vector<size_t> m_Shape;
    size_t m_Size;

public: // --- STATIC ---
    static Matrix zeros(const std::vector<size_t>& shape = {});
    static Matrix zeros_like(const Matrix& t);

    static Matrix ones(const std::vector<size_t>& shape = {});
    static Matrix ones_like(const Matrix& t);

    static Matrix eye(size_t dim);
    static Matrix linspace(float start, float end, size_t samples = 50);

    static std::default_random_engine rng;

    static void seed(int s);

    static Matrix normal(const std::vector<size_t>& shape = {}, float mu = 0.0f,
                         float sigma = 1.0f);

    static Matrix uniform(const std::vector<size_t>& shape = {}, float a = 0.0f,
                          float b = 1.0f);
};

template <>
struct std::formatter<Matrix>
{
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    auto format(const Matrix& t, std::format_context& ctx) const
    {
        auto out = ctx.out();

        const auto& shape = t.shape();
        const float* data = t.data();

        // Scalar
        if (t.is_scalar())
            return std::format_to(out, "{:>10.3f}", data[0]);

        // Vector
        if (t.ndim() == 1)
        {
            std::format_to(out, "[");
            for (size_t i = 0; i < shape[0]; ++i)
            {
                if (i > 0)
                    std::format_to(out, " ");

                std::format_to(out, "{:>10.3f}", data[i]);
            }

            return std::format_to(out, "]");
        }

        // Matrix
        const size_t rows = shape[0];
        const size_t cols = shape[1];

        std::format_to(out, "[");

        for (size_t i = 0; i < rows; ++i)
        {
            if (i > 0)
                std::format_to(out, " ");

            std::format_to(out, "[");

            for (size_t j = 0; j < cols; ++j)
            {
                if (j > 0)
                    std::format_to(out, " ");

                std::format_to(out, "{:>10.3f}", data[i * cols + j]);
            }

            std::format_to(out, "]");

            if (i + 1 < rows)
                std::format_to(out, "\n");
        }

        return std::format_to(out, "]");
    }
};

template <typename Op>
Matrix unary_operator(const Matrix& x, Op op)
{
    float* data = new float[x.size()];
    for (size_t i = 0; i < x.size(); ++i)
        data[i] = op(x.m_Data[i]);

    return Matrix(data, x.m_Shape);
}

template <typename Op>
Matrix binary_operator(const Matrix& a, const Matrix& b, Op op)
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

    return Matrix(data, a.is_scalar() ? b.shape() : a.shape());
}

template <typename Op>
Matrix& binary_assign_operator(Matrix& a, const Matrix& b, Op op)
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

#endif
