#include "metrics.hpp"

Matrix accuracy_score(const std::vector<Tensor>& predicted,
                      const std::vector<Tensor>& target)
{
    Matrix right = 0;
    for (size_t i = 0; i < predicted.size(); i++)
        right += sum(predicted[i].item() == target[i].item());

    return right / predicted.size();
}
