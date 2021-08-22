/*
* FeedForwardNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/NeuralNetwork.h>

template <typename Node, typename Edge>
class FeedForwardNetwork : public NeuralNetwork<Node, Edge>
{
public:
    // Type Declarations
    using Base = NeuralNetwork<Node, Edge>;
    using Nodes = Base::Nodes;
    using Edges = Base::Edges;
    using NodeIds = Base::NodeIds;
    using EdgeIds = Base::EdgeIds;
    using NodeData = Base::NodeData;
    using NodeDatas = Base::NodeDatas;

    // Constructor from network information
    FeedForwardNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& inputNodes, const NodeIds& outputNodes);

    // Create a copy of this network.
    virtual auto clone() const->std::shared_ptr<NeuralNetwork<Node, Edge>> override;

    // Return type of this network.
    virtual NeuralNetworkType getType() const override { return NeuralNetworkType::FEED_FORWARD; }

    // Feed forward network shouldn't have circular connections.
    virtual bool allowsCircularNetwork() const override { return false; }

    virtual bool validate() const;

protected:
    // Return true if a new edge can be added between inNode and outNode.
    virtual bool canAddEdgeAt(NodeId inNode, NodeId outNode) const override;

    // Recursive functions used internally.
    bool canAddEdgeAtRecursive(NodeId outNode, NodeId curNode) const;
};

template <typename Node, typename Edge>
FeedForwardNetwork<Node, Edge>::FeedForwardNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& inputNodes, const NodeIds& outputNodes)
    : Base(nodes, edges, inputNodes, outputNodes)
{
}

template <typename Node, typename Edge>
auto FeedForwardNetwork<Node, Edge>::clone() const->std::shared_ptr<NeuralNetwork<Node, Edge>>
{
    return std::make_shared<FeedForwardNetwork<Node, Edge>>(*this);
}

template <typename Node, typename Edge>
bool FeedForwardNetwork<Node, Edge>::canAddEdgeAtRecursive(NodeId outNode, NodeId curNode) const
{
    const NodeData& node = this->getNodeData(curNode);

    if (node.getIncomingEdges().empty())
    {
        return true;
    }

    // Follow edges backward and check we never see outNode.
    for (EdgeId e : node.getIncomingEdges())
    {
        NodeId n = this->getInNode(e);
        if (n == outNode || !canAddEdgeAtRecursive(outNode, n))
        {
            return false;
        }
    }

    return true;
}

template <typename Node, typename Edge>
bool FeedForwardNetwork<Node, Edge>::canAddEdgeAt(NodeId inNode, NodeId outNode) const
{
    for (NodeId id : this->m_inputNodes)
    {
        if (id == outNode)
        {
            WARN("Input node cannot have an incoming edge. Abort adding a new edge.");
            return false;
        }
    }

    for (NodeId id : this->m_outputNodes)
    {
        if (id == inNode)
        {
            WARN("Output node cannot have an outgoing edge. Abort adding a new edge.");
            return false;
        }
    }

    return canAddEdgeAtRecursive(outNode, inNode);
}

template <typename Node, typename Edge>
bool FeedForwardNetwork<Node, Edge>::validate() const
{
#ifdef DEBUG_SLOW
    if (!this->Base::validate())
    {
        return false;
    }

    if (this->m_inputNodes.size() < 1) return false;
    if (this->m_outputNodes.size() < 1) return false;

    // Validate all input nodes.
    {
        std::unordered_set<NodeId> nodes;
        for (NodeId n : this->m_inputNodes)
        {
            if (!this->hasNode(n)) return false;

            // Make sure that the id is unique.
            if (nodes.find(n) != nodes.end()) return false;
            nodes.insert(n);

            // Make sure that no edge has this node as its outNode.
            for (const auto& edge : this->m_edges)
            {
                const Edge& e = edge.m_edge;
                if (e.getOutNode() == n) return false;
            }
        }
    }

    // Validate all output nodes.
    {
        std::unordered_set<NodeId> nodes;
        for (NodeId n : this->m_outputNodes)
        {
            if (!this->hasNode(n)) return false;

            // Make sure that the id is unique.
            if (nodes.find(n) != nodes.end()) return false;
            nodes.insert(n);

            if (this->getIncomingEdges(n).empty()) return false;

            // Make sure that no edge has this node as its inNode.
            for (const auto& edge : this->m_edges)
            {
                const Edge& e = edge.m_edge;
                if (e.getInNode() == n) return false;
            }
        }
    }

    // Make sure the the network doesn't contain circular edges.
    if (hasCircularEdges()) return false;
#endif
    return true;
}
