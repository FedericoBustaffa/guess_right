#include <chrono>
#include <cstdlib>
#include <print>

#include "activations.hpp"
#include "computational_graph.hpp"
#include "data.hpp"
#include "losses.hpp"
#include "modules.hpp"
#include "ndarray.hpp"
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
    const size_t input_dim = 1;
    const size_t output_dim = 1;
    NDArray slope = NDArray::normal({input_dim, output_dim}, 1, 1);
    NDArray intercept = NDArray::normal({output_dim}, 2, 0.1);

    // training set
    std::vector<Tensor> x_train = random(256, {input_dim}, 0, 1);
    std::vector<Tensor> y_train;
    for (size_t i = 0; i < x_train.size(); i++)
        y_train.push_back(matmul(slope.transpose(), x_train[i]) + intercept);
    Dataset train(x_train, y_train);
    Loader train_loader(train, 32, false);

    // test set
    std::vector<Tensor> x_test = random(32, {input_dim}, 0, 1);
    std::vector<Tensor> y_test;
    for (size_t i = 0; i < x_test.size(); i++)
        y_test.push_back(matmul(slope.transpose(), x_test[i]) + intercept);
    Dataset test(x_test, y_test);
    Loader test_loader(test, 32, false);

    // model definition
    Model model;
    size_t hidden_dim = 64;
    model.add<Linear>(input_dim, hidden_dim);
    model.add<ReLU>();
    model.add<Linear>(hidden_dim, hidden_dim);
    model.add<ReLU>();
    model.add<Linear>(hidden_dim, output_dim);

    StochasticGradientDescent sgd(model.parameters(), 1e-2);
    MeanSquaredError loss_fn;

    // tranining loop
    double training_time = 0.0;
    double loader_time = 0.0;
    double forward_time = 0.0;
    double backpropagation_time = 0.0;
    double sgd_time = 0.0;

    const size_t max_epochs = 200;
    std::vector<float> train_history;
    train_history.reserve(max_epochs);

    std::vector<float> test_history;
    test_history.reserve(max_epochs);

    for (size_t e = 0; e < max_epochs; e++)
    {
        for (size_t i = 0; i < train_loader.size(); i++)
        {
            auto training_start = std::chrono::high_resolution_clock::now();

            // fetch a batch from the dataloader
            auto loader_start = std::chrono::high_resolution_clock::now();
            Dataset batch = train_loader.get();
            std::vector<Tensor> x_batch = batch.inputs();
            std::vector<Tensor> y_batch = batch.targets();
            auto loader_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> loader_duration =
                loader_end - loader_start;
            loader_time += loader_duration.count();

            // reset gradients to zero
            sgd.zero_grad();

            // forward pass
            auto forward_start = std::chrono::high_resolution_clock::now();
            std::vector<Tensor> pred = model(x_batch);

            // compute loss
            Tensor loss = loss_fn(pred, y_batch);
            auto forward_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> forward_duration =
                forward_end - forward_start;
            forward_time += forward_duration.count();

            // compute gradients
            auto backward_start = std::chrono::high_resolution_clock::now();
            loss.backward();
            auto backward_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> backward_duration =
                backward_end - backward_start;
            backpropagation_time += backward_duration.count();

            // update parameters
            auto sgd_start = std::chrono::high_resolution_clock::now();
            sgd.step();
            auto sgd_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> sgd_duration = sgd_end - sgd_start;
            sgd_time += sgd_duration.count();

            auto training_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> trainining_duration =
                training_end - training_start;
            training_time += trainining_duration.count();

            std::print("epoch {:>4}, batch {}/{},  loss: {}\r", e + 1, i + 1,
                       train_loader.size(), loss.item());
        }
        if ((e + 1) % 50 == 0)
            std::println();

        {
            NoGrad guard;

            Tensor train_loss = loss_fn(model(x_train), y_train);
            train_history.push_back((float)train_loss.item());

            Tensor test_loss = loss_fn(model(x_test), y_test);
            test_history.push_back((float)test_loss.item());
        }
    }

    // benchmark results
    std::print("training time: {:.4f} seconds\n", training_time);

    std::print("loader time: {:.4f} seconds -> {:.2f}%\n", loader_time,
               loader_time / training_time * 100.0);

    std::print("forward time: {:.4f} seconds -> {:.2f}%\n", forward_time,
               forward_time / training_time * 100.0);

    std::print("backward time: {:.4f} seconds -> {:.2f}%\n",
               backpropagation_time,
               backpropagation_time / training_time * 100.0);

    std::print("optimizer time: {:.4f} seconds -> {:.2f}%\n", sgd_time,
               sgd_time / training_time * 100.0);

    std::println("final training loss: {:>6.4f}", train_history.back());
    std::println("final test loss: {:>6.4f}", test_history.back());

    return 0;
}
