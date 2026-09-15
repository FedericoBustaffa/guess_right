#ifndef NODE_HPP
#define NODE_HPP

#include <memory>
#include <vector>

#include "ndarray.hpp"

enum class Operation
{
    Nop,
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Transpose,
    Sum,
    Mean,
    Product,
    Matmul,
    Pow,
    Log,
    Exp,
    Sin,
    Cos,
    Tanh,
    Maximum,
    Minimum
};

struct Node
{
    Node(const NDArray& data, bool requires_grad, Operation operation,
         std::vector<std::shared_ptr<Node>>&& parents);

    void backward();

    ~Node();

    NDArray data;
    NDArray grad;
    bool requires_grad;
    Operation operation;
    std::vector<std::shared_ptr<Node>> parents;
};

#endif
