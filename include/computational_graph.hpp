#ifndef COMPUTATIONAL_GRAPH_HPP
#define COMPUTATIONAL_GRAPH_HPP

#include <memory>

#include "node.hpp"

class ComputationalGraph
{
public:
    ComputationalGraph(const ComputationalGraph& other) = delete;
    ComputationalGraph(ComputationalGraph&& other) = delete;

    static void backward(const std::shared_ptr<Node>& node);

    static void set_no_grad(bool no_grad);

    static bool no_grad();

    ~ComputationalGraph();

private:
    ComputationalGraph();

private:
    static ComputationalGraph s_Instance;
    static bool s_NoGrad;
};

class NoGrad
{
public:
    NoGrad() : m_Previous(ComputationalGraph::no_grad())
    {
        ComputationalGraph::set_no_grad(true);
    }

    ~NoGrad() { ComputationalGraph::set_no_grad(m_Previous); }

private:
    bool m_Previous;
};
#endif
