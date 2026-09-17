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
    std::vector<Tensor> x_train = random(128, {input_dim}, 0, 1);
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

    // model definition
    Sequential model;
    size_t hidden_dim = 32;
    model.add<Linear>(input_dim, hidden_dim);
    model.add<ReLU>();
    model.add<Linear>(hidden_dim, hidden_dim);
    model.add<ReLU>();
    model.add<Linear>(hidden_dim, output_dim);

    StochasticGradientDescent sgd(model.parameters(), 1e-2);
    MeanSquaredError loss_fn;

    const size_t max_epochs = 200;
    std::vector<float> train_history;
    train_history.reserve(max_epochs);

    std::vector<float> test_history;
    test_history.reserve(max_epochs);

    // training loop
    for (size_t e = 0; e < max_epochs; e++)
    {
        // batch processing
        for (size_t i = 0; i < train_loader.size(); i++)
        {
            // fetch a batch from the dataloader
            Dataset batch = train_loader.get();
            std::vector<Tensor> x_batch = batch.inputs();
            std::vector<Tensor> y_batch = batch.targets();

            // reset gradients to zero
            sgd.zero_grad();

            // forward pass
            std::vector<Tensor> pred = model(x_batch);

            // compute loss
            Tensor loss = loss_fn(pred, y_batch);

            // compute gradients
            loss.backward();

            // update parameters
            sgd.step();

            std::print("epoch {:>4}, batch {}/{},  loss: {:>6.4f}\r", e + 1,
                       i + 1, train_loader.size(), (float)loss.item());
        }
        if ((e + 1) % 50 == 0)
            std::println();

        { // final prediction on the full dataset and loss history recording
            NoGrad guard;

            Tensor train_loss = loss_fn(model(x_train), y_train);
            train_history.push_back((float)train_loss.item());

            Tensor test_loss = loss_fn(model(x_test), y_test);
            test_history.push_back((float)test_loss.item());
        }
    }

    std::println("final training loss: {:>6.4f}", train_history.back());
    std::println("final test loss: {:>6.4f}", test_history.back());

    return 0;
}
