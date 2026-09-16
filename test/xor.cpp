#include <cmath>
#include <print>

#include "activations.hpp"
#include "computational_graph.hpp"
#include "losses.hpp"
#include "metrics.hpp"
#include "modules.hpp"
#include "ndarray.hpp"
#include "optimizer.hpp"

int main(int argc, const char** argv)
{
    const size_t input_dim = 2;
    const size_t output_dim = 1;

    std::vector<Tensor> x;
    x.emplace_back(NDArray({0.0f, 0.0f}, {input_dim}), false);
    x.emplace_back(NDArray({0.0f, 1.0f}, {input_dim}), false);
    x.emplace_back(NDArray({1.0f, 0.0f}, {input_dim}), false);
    x.emplace_back(NDArray({1.0f, 1.0f}, {input_dim}), false);

    std::vector<Tensor> y;
    y.emplace_back(NDArray(0.0f), false);
    y.emplace_back(NDArray(1.0f), false);
    y.emplace_back(NDArray(1.0f), false);
    y.emplace_back(NDArray(0.0f), false);

    Sequential model;
    const size_t hidden_dim = 8;
    model.add<Linear>(input_dim, hidden_dim);
    model.add<LeakyReLU>();
    model.add<Linear>(hidden_dim, output_dim);
    model.add<Sigmoid>();

    StochasticGradientDescent sgd(model.parameters(), 1e-2);
    BinaryCrossEntropy loss_fn;

    const size_t batch_size = 1;
    const size_t n_batches = (x.size() + batch_size - 1) / batch_size;

    for (size_t e = 0; e < 1000; e++)
    {
        // NDArray epoch_loss = 0.0f;
        for (size_t i = 0; i < n_batches; i++)
        {
            NDArray batch_loss = 0.0f;
            const size_t batch_start = i * batch_size;
            const size_t batch_end =
                std::min(batch_start + batch_size, x.size());
            const size_t actual_batch_size = batch_end - batch_start;
            sgd.zero_grad();

            for (size_t j = batch_start; j < batch_end; j++)
            {
                // forward pass
                Tensor pred = model(x[j]);

                // compute loss
                Tensor loss = loss_fn(pred, y[j]) / actual_batch_size;
                batch_loss += loss.item();

                // compute gradients
                loss.backward();
            }

            sgd.step();
            // epoch_loss += batch_loss;

            std::print("epoch {:>4} batch {:>3}/{:<3} loss: {}\r", e + 1, i + 1,
                       n_batches, batch_loss);
        }
        if ((e + 1) % 100 == 0)
            std::println();
    }

    {
        NoGrad nograd;
        std::vector<Tensor> predictions;
        for (size_t i = 0; i < x.size(); i++)
        {
            auto pred = model(x[i]);
            predictions.push_back(round(pred));
            std::print("target: {} output: {}\n", y[i].item(), pred.item());
        }

        std::print("accuracy: {}\n", accuracy_score(predictions, y));
    }

    return 0;
}
