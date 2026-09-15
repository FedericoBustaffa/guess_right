#include "metrics.hpp"

NDArray accuracy_score(const std::vector<Tensor>& predicted,
                       const std::vector<Tensor>& target)
{
    NDArray right = 0;
    for (size_t i = 0; i < predicted.size(); i++)
        right += sum(predicted[i].item() == target[i].item());

    return right / predicted.size();
}
