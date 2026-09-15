#ifndef METRICS_HPP
#define METRICS_HPP

#include "tensor.hpp"

NDArray accuracy_score(const std::vector<Tensor>& predicted,
                       const std::vector<Tensor>& target);

#endif
