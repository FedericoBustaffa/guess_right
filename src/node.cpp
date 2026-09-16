#include "node.hpp"
#include "ndarray.hpp"

Node::Node(const NDArray& data, bool requires_grad, Operation operation,
           std::vector<std::shared_ptr<Node>>&& parents)
    : data(data), grad(NDArray::zeros_like(data)), requires_grad(requires_grad),
      operation(operation), parents(std::move(parents))
{
}

Node::Node(NDArray&& data, bool requires_grad, Operation operation,
           std::vector<std::shared_ptr<Node>>&& parents)
    : data(std::move(data)), grad(NDArray::zeros_like(this->data)),
      requires_grad(requires_grad), operation(operation),
      parents(std::move(parents))
{
}

void Node::backward()
{
    switch (operation)
    {
    case Operation::Nop: {
        return;
    }

    case Operation::Neg: {
        const std::shared_ptr<Node>& a = parents[0];
        if (a->requires_grad)
            a->grad -= grad;

        break;
    }

    case Operation::Add: {
        const std::shared_ptr<Node>& a = parents[0];
        const std::shared_ptr<Node>& b = parents[1];
        if (a->requires_grad)
            a->grad += a->data.is_scalar() ? sum(grad) : grad;

        if (b->requires_grad)
            b->grad += b->data.is_scalar() ? sum(grad) : grad;
        break;
    }

    case Operation::Sub: {
        const std::shared_ptr<Node>& a = parents[0];
        const std::shared_ptr<Node>& b = parents[1];

        if (a->requires_grad)
            a->grad += a->data.is_scalar() ? sum(grad) : grad;

        if (b->requires_grad)
            b->grad -= b->data.is_scalar() ? sum(grad) : grad;
        break;
    }

    case Operation::Mul: {
        const std::shared_ptr<Node>& a = parents[0];
        const std::shared_ptr<Node>& b = parents[1];

        if (a->requires_grad)
            a->grad +=
                a->data.is_scalar() ? sum(grad * b->data) : grad * b->data;

        if (b->requires_grad)
            b->grad +=
                b->data.is_scalar() ? sum(grad * a->data) : grad * a->data;
        break;
    }

    case Operation::Div: {
        const std::shared_ptr<Node>& a = parents[0];
        const std::shared_ptr<Node>& b = parents[1];

        if (a->requires_grad)
            a->grad +=
                a->data.is_scalar() ? sum(grad / b->data) : grad / b->data;

        if (b->requires_grad)
            b->grad -= b->data.is_scalar()
                           ? sum(grad * a->data / pow(b->data, 2.0f))
                           : grad * a->data / pow(b->data, 2.0f);
        break;
    }

    case Operation::Transpose: {
        const std::shared_ptr<Node>& a = parents[0];

        if (a->requires_grad)
            a->grad += grad.transpose();
        break;
    }

    case Operation::Sum: {
        const std::shared_ptr<Node>& a = parents[0];

        if (a->requires_grad)
            a->grad += grad;
        break;
    }

    case Operation::Mean: {
        const std::shared_ptr<Node>& a = parents[0];
        if (a->requires_grad)
            a->grad += grad / static_cast<float>(a->data.size());
        break;
    }

    case Operation::Product: {
        const auto& a = parents[0];

        if (a->requires_grad)
            a->grad += grad * (data / a->data);
        break;
    }

    case Operation::Matmul: {
        const std::shared_ptr<Node>& a = parents[0];
        const std::shared_ptr<Node>& b = parents[1];

        if (a->requires_grad)
            a->grad += matmul(grad, b->data.transpose());

        if (b->requires_grad)
            b->grad += matmul(a->data.transpose(), grad);
        break;
    }

    case Operation::Pow: {
        const std::shared_ptr<Node>& a = parents[0];
        const std::shared_ptr<Node>& b = parents[1];

        if (a->requires_grad)
            a->grad += a->data.is_scalar()
                           ? sum(grad * b->data * pow(a->data, b->data - 1.0f))
                           : grad * b->data * pow(a->data, b->data - 1.0f);

        if (b->requires_grad)
            b->grad += b->data.is_scalar() ? sum(grad * data * log(a->data))
                                           : grad * data * log(a->data);
        break;
    }

    case Operation::Log: {
        const std::shared_ptr<Node>& a = parents[0];
        if (a->requires_grad)
            a->grad += grad / a->data;
        break;
    }

    case Operation::Exp: {
        const std::shared_ptr<Node>& a = parents[0];
        if (a->requires_grad)
            a->grad += grad * data;
        break;
    }

    case Operation::Sin: {
        const std::shared_ptr<Node>& a = parents[0];
        if (a->requires_grad)
            a->grad += grad * cos(data);
        break;
    }

    case Operation::Cos: {
        const std::shared_ptr<Node>& a = parents[0];
        if (a->requires_grad)
            a->grad -= grad * sin(data);
        break;
    }

    case Operation::Tanh: {
        const std::shared_ptr<Node>& a = parents[0];
        if (a->requires_grad)
            a->grad += grad * (1.0f - pow(data, 2.0f));
        break;
    }

    case Operation::Maximum: {
        const std::shared_ptr<Node>& a = parents[0];
        const std::shared_ptr<Node>& b = parents[1];

        if (a->requires_grad)
            a->grad += a->data.is_scalar() ? sum(grad * (a->data > b->data))
                                           : grad * (a->data > b->data);

        if (b->requires_grad)
            b->grad += b->data.is_scalar() ? sum(grad * (b->data > a->data))
                                           : grad * (b->data > a->data);
        break;
    }

    case Operation::Minimum: {
        const std::shared_ptr<Node>& a = parents[0];
        const std::shared_ptr<Node>& b = parents[1];

        if (a->requires_grad)
            a->grad += a->data.is_scalar() ? sum(grad * (a->data < b->data))
                                           : grad * (a->data < b->data);

        if (b->requires_grad)
            b->grad += b->data.is_scalar() ? sum(grad * (b->data < a->data))
                                           : grad * (b->data < a->data);
        break;
    }
    }
}

Node::~Node() {}
