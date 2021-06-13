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

    // Constructors

    // Constructor from network information
    FeedForwardNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& inputNodes, const NodeIds& outputNodes);

    // Create a copy of this network.
    virtual auto clone() const->std::shared_ptr<NeuralNetwork<Node, Edge>> override;

    // Feed forward network shouldn't have circular connections.
    virtual bool allowsCircularNetwork() const override { return false; }

    // Return true if this network has circular connections.
    bool hasCircularEdges() const;

    // Evaluates this network and calculate new values for each node.
    virtual void evaluate() override;

    virtual bool validate() const;

protected:
    virtual bool canAddEdgeAt(NodeId inNode, NodeId outNode) const override;

    bool canAddEdgeAtRecursive(NodeId outNode, NodeId curNode) const;

    // Data used evaluation
    struct EvaluationData
    {
        enum class NodeState
        {
            NONE,
            EVALUATED
        };

        EvaluationData(const FeedForwardNetwork* network);

        inline NodeState getNodeState(NodeId id) const { return m_nodeStates[m_id2Index.at(id)]; }
        inline void setNodeState(NodeId id, NodeState state) { m_nodeStates[m_id2Index.at(id)] = state; }

        std::unordered_map<NodeId, int> m_id2Index; // Map between NodeId and its index in m_nodeStates.
        std::vector<NodeState> m_nodeStates; // Status of each node.
    };

    void evaluateNodeRecursive(NodeId id, EvaluationData& data);

    bool hasCircularEdgesRecursive(NodeId id, std::unordered_set<NodeId> visitedNodes) const;
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
    const NodeData& node = this->m_nodes.at(curNode);

    if (node.m_incomingEdges.empty())
    {
        return true;
    }

    for (EdgeId e : node.m_incomingEdges)
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
void FeedForwardNetwork<Node, Edge>::evaluate()
{
    assert(validate());

    // Initialize evaluation data
    EvaluationData data(this);

    // Evaluate output nodes
    for (NodeId id : this->m_outputNodes)
    {
        evaluateNodeRecursive(id, data);
    }
}

template <typename Node, typename Edge>
void FeedForwardNetwork<Node, Edge>::evaluateNodeRecursive(NodeId id, EvaluationData& data)
{
    NodeData& node = this->m_nodes[id];

    if (node.m_incomingEdges.empty())
    {
        // This is input/bias node. Don't update the value.
        return;
    }

    // Calculate value of this node by visiting all parent nodes.
    float sumValue = 0;
    for (EdgeId incomingId : node.m_incomingEdges)
    {
        if (this->getWeight(incomingId) == 0.f)
        {
            continue;
        }

        NodeId inNodeId = this->getInNode(incomingId);

        // Recurse if we haven't evaluated this parent node yet.
        // NOTE: We assume that the network doesn't have any circular edges.
        if (data.getNodeState(inNodeId) != EvaluationData::NodeState::EVALUATED)
        {
            this->evaluateNodeRecursive(inNodeId, data);
        }

        // Add a value from this parent.
        sumValue += this->getNode(inNodeId).getValue() * this->getWeight(incomingId);
    }

    data.setNodeState(id, EvaluationData::NodeState::EVALUATED);
    node.m_node.setValue(sumValue);
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
            for (const auto& itr : this->m_edges)
            {
                const Edge& e = itr.second;
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
            for (const auto& itr : this->m_edges)
            {
                const Edge& e = itr.second;
                if (e.getInNode() == n) return false;
            }
        }
    }

    // Make sure the the network doesn't contain circular edges.
    if (hasCircularEdges()) return false;
#endif
    return true;
}

template <typename Node, typename Edge>
bool FeedForwardNetwork<Node, Edge>::hasCircularEdgesRecursive(NodeId id, std::unordered_set<NodeId> visitedNodes) const
{
    if (visitedNodes.find(id) != visitedNodes.end())
    {
        // Already visited this node.
        return true;
    }

    visitedNodes.insert(id);

    for (EdgeId e : this->getIncomingEdges(id))
    {
        if (hasCircularEdgesRecursive(this->m_edges.at(e).getInNode(), visitedNodes))
        {
            return true;
        }
    }

    return false;
}

template <typename Node, typename Edge>
bool FeedForwardNetwork<Node, Edge>::hasCircularEdges() const
{
    std::unordered_set<NodeId> checkedNodes;

    for (const auto& itr : this->m_nodes)
    {
        NodeId id = itr.first;

        if (checkedNodes.find(id) != checkedNodes.end())
        {
            // This node is already checked.
            continue;
        }

        // Check if this node is a part of circular links.
        std::unordered_set<NodeId> visitedNodes;
        if (hasCircularEdgesRecursive(id, visitedNodes)) return true;

        // Concatenate visited nodes to checked nodes.
        for (NodeId n : visitedNodes)
        {
            checkedNodes.insert(n);
        }
    }

    return false;
}

template <typename Node, typename Edge>
FeedForwardNetwork<Node, Edge>::EvaluationData::EvaluationData(const FeedForwardNetwork* network)
{
    const int numNodes = network->getNumNodes();
    m_id2Index.reserve(numNodes);
    int counter = 0;
    for (const Base::NodeEntry& node : network->getNodes())
    {
        m_id2Index[node.first] = counter++;
    }

    m_nodeStates.resize(numNodes, NodeState::NONE);
}