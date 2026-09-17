#include "computational_graph.hpp"
#include "ndarray.hpp"

#include <functional>
#include <unordered_set>

ComputationalGraph ComputationalGraph::s_Instance; // singletone instance

bool ComputationalGraph::s_NoGrad = false;

ComputationalGraph::ComputationalGraph() {};

void ComputationalGraph::backward(const std::shared_ptr<Node>& node)
{
    if (!node->requires_grad)
        return;

    node->grad = NDArray::ones_like(node->data);

    std::vector<Node*> order;
    std::unordered_set<Node*> visited;

    std::function<void(Node*)> dfs = [&](Node* current) {
        if (!visited.insert(current).second)
            return;

        for (const std::shared_ptr<Node>& p : current->parents)
            dfs(p.get());

        order.push_back(current);
    };

    dfs(node.get());

    for (auto it = order.rbegin(); it != order.rend(); ++it)
        (*it)->backward();
}

void ComputationalGraph::set_no_grad(bool no_grad) { s_NoGrad = no_grad; }

bool ComputationalGraph::no_grad() { return s_NoGrad; }

ComputationalGraph::~ComputationalGraph() {}
