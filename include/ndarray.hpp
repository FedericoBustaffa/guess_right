#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <cstddef>
#include <format>
#include <random>
#include <vector>

class NDArray
{
public:
    NDArray(float data);
    NDArray(float* data, const std::vector<size_t>& shape);
    NDArray(const std::vector<float>& data, const std::vector<size_t>& shape);
    NDArray(const NDArray& other);
    NDArray(NDArray&& other);

    inline const float* data() const { return m_Data; }
    inline const std::vector<size_t>& shape() const { return m_Shape; }
    inline size_t ndim() const { return m_Shape.size(); }
    inline size_t size() const { return m_Size; }
    inline bool is_scalar() const { return m_Shape[0] == 1 && m_Shape[1] == 1; }

    explicit operator float() const;

    //  Operators
    NDArray operator-() const;
    friend NDArray operator+(const NDArray& a, const NDArray& b);
    friend NDArray operator-(const NDArray& a, const NDArray& b);
    friend NDArray operator*(const NDArray& a, const NDArray& b);
    friend NDArray operator/(const NDArray& a, const NDArray& b);

    // Assign operators
    NDArray& operator=(const NDArray& other);
    NDArray& operator=(NDArray&& other) noexcept;
    NDArray& operator+=(const NDArray& other);
    NDArray& operator-=(const NDArray& other);
    NDArray& operator*=(const NDArray& other);
    NDArray& operator/=(const NDArray& other);

    // Logic operators
    friend NDArray operator<(const NDArray& a, const NDArray& b);
    friend NDArray operator<=(const NDArray& a, const NDArray& b);
    friend NDArray operator==(const NDArray& a, const NDArray& b);
    friend NDArray operator>=(const NDArray& a, const NDArray& b);
    friend NDArray operator>(const NDArray& a, const NDArray& b);

    // functions
    NDArray transpose() const;
    friend NDArray sum(const NDArray& a);
    friend NDArray mean(const NDArray& a);
    friend NDArray product(const NDArray& a);
    friend NDArray matmul(const NDArray& a, const NDArray& b);
    friend NDArray pow(const NDArray& x, const NDArray& e);
    friend NDArray abs(const NDArray& x);
    friend NDArray sqrt(const NDArray& x);
    friend NDArray log(const NDArray& x);
    friend NDArray exp(const NDArray& x);
    friend NDArray sin(const NDArray& x);
    friend NDArray cos(const NDArray& x);
    friend NDArray tanh(const NDArray& x);
    friend NDArray maximum(const NDArray& a, const NDArray& b);
    friend NDArray minimum(const NDArray& a, const NDArray& b);

    friend NDArray round(const NDArray& a);

    ~NDArray();

private:
    template <typename Op>
    friend NDArray unary_operator(const NDArray& x, Op op);

    template <typename Op>
    friend NDArray binary_operator(const NDArray& a, const NDArray& b, Op op);

    template <typename Op>
    friend NDArray& binary_assign_operator(NDArray& a, const NDArray& b, Op op);

private:
    float* m_Data;
    std::vector<size_t> m_Shape;
    size_t m_Size;

public: // --- STATIC ---
    static NDArray zeros(const std::vector<size_t>& shape = {});
    static NDArray zeros_like(const NDArray& t);

    static NDArray ones(const std::vector<size_t>& shape = {});
    static NDArray ones_like(const NDArray& t);

    static NDArray eye(size_t dim);
    static NDArray linspace(float start, float end, size_t samples = 50);

    static std::default_random_engine rng;

    static void seed(int s);

    static NDArray normal(const std::vector<size_t>& shape = {},
                          float mu = 0.0f, float sigma = 1.0f);

    static NDArray uniform(const std::vector<size_t>& shape = {},
                           float a = 0.0f, float b = 1.0f);
};

template <>
struct std::formatter<NDArray>
{
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    auto format(const NDArray& t, std::format_context& ctx) const
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
NDArray unary_operator(const NDArray& x, Op op)
{
    float* data = new float[x.size()];
    for (size_t i = 0; i < x.size(); ++i)
        data[i] = op(x.m_Data[i]);

    return NDArray(data, x.m_Shape);
}

template <typename Op>
NDArray binary_operator(const NDArray& a, const NDArray& b, Op op)
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

template <typename Op>
NDArray& binary_assign_operator(NDArray& a, const NDArray& b, Op op)
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
