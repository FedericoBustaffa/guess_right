#include <print>

#include "activations.hpp"
#include "computational_graph.hpp"
#include "losses.hpp"
#include "modules.hpp"
#include "optimizer.hpp"

std::vector<Tensor> random(size_t n, const std::vector<size_t>& shape, float mu,
                           float sigma)
{
    std::vector<Tensor> out;
    for (size_t i = 0; i < n; i++)
        out.emplace_back(NDArray::normal(shape, mu, sigma));

    return out;
}

int main(int argc, const char** argv)
{
    const size_t input_dim = 12;
    const size_t output_dim = 4;
    NDArray slope = NDArray::normal({input_dim, output_dim}, 1, 1);
    NDArray intercept = NDArray::normal({output_dim}, 2, 0.1);

    std::vector<Tensor> x_train = random(256, {input_dim}, 0, 1);
    std::vector<Tensor> y_train;
    for (size_t i = 0; i < x_train.size(); i++)
        y_train.push_back(matmul(slope.transpose(), x_train[i]) + intercept);

    std::vector<Tensor> x_test = random(32, {input_dim}, 0, 1);
    std::vector<Tensor> y_test;
    for (size_t i = 0; i < x_test.size(); i++)
        y_test.push_back(matmul(slope.transpose(), x_test[i]) + intercept);

    Sequential model;
    size_t hidden_dim = 16;
    model.add<Linear>(input_dim, hidden_dim);
    model.add<ReLU>();

    Residual skip;
    skip.add<Linear>(hidden_dim, hidden_dim);
    skip.add<ReLU>();
    model.add<Residual>(skip);

    model.add<Linear>(hidden_dim, output_dim);

    StochasticGradientDescent sgd(model.parameters(), 1e-2);
    SquaredError loss_fn;

    const size_t batch_size = 32;
    const size_t n_batches = (x_train.size() + batch_size - 1) / batch_size;

    for (size_t e = 0; e < 200; e++)
    {
        // NDArray epoch_loss = 0.0f;
        for (size_t i = 0; i < n_batches; i++)
        {
            NDArray batch_loss = 0.0f;
            const size_t batch_start = i * batch_size;
            const size_t batch_end =
                std::min(batch_start + batch_size, x_train.size());
            const size_t actual_batch_size = batch_end - batch_start;
            sgd.zero_grad();

            for (size_t j = batch_start; j < batch_end; j++)
            {
                // forward pass
                Tensor pred = model(x_train[j]);

                // compute loss
                Tensor loss = loss_fn(pred, y_train[j]) / actual_batch_size;
                batch_loss += loss.item();

                // compute gradients
                loss.backward();
            }

            sgd.step();
            // epoch_loss += batch_loss;

            std::print("epoch {:>4} batch {:>3}/{:<3} loss: {}\r", e + 1, i + 1,
                       n_batches, batch_loss);
        }
        if ((e + 1) % 50 == 0)
            std::println();
    }

    {
        NoGrad nograd;
        NDArray total_loss = 0.0f;
        for (size_t i = 0; i < x_train.size(); i++)
        {
            // forward pass
            Tensor pred = model(x_train[i]);

            // compute loss
            Tensor loss = loss_fn(pred, y_train[i]) / x_train.size();
            total_loss += loss.item();
        }
        std::println("final training loss: {}", total_loss);
    }

    {
        NoGrad nograd;
        NDArray total_loss = 0.0f;
        for (size_t i = 0; i < x_test.size(); i++)
        {
            // forward pass
            Tensor pred = model(x_test[i]);

            // compute loss
            Tensor loss = loss_fn(pred, y_test[i]) / x_test.size();
            total_loss += loss.item();
        }
        std::println("final test loss: {}", total_loss);
    }

    return 0;
}
