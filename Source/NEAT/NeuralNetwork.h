/*
* NeuralNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>

#include <Common/BaseType.h>

// Declarations of types.
DECLARE_ID(NodeId);
DECLARE_ID(EdgeId);

// Base struct of node.
struct NodeBase
{
    using EdgeIds = std::vector<EdgeId>;

    virtual float getValue() const = 0;
    virtual void setValue(float value) = 0;
    virtual EdgeIds getIncomingEdges() const = 0;
};

// Base struct of edge.
struct EdgeBase
{
    virtual NodeId getInNode() const = 0;
    virtual NodeId getOutNode() const = 0;
    virtual float getWeight() const = 0;
    virtual void setWeight(float weight) = 0;
};

template <typename Node, typename Edge>
class NeuralNetwork
{
public:
    // Declarations of types.
    using Nodes = std::unordered_map<NodeId, Node>;
    using Edges = std::unordered_map<EdgeId, Edge>;
    using NodeIds = std::vector<NodeId>;
    using EdgeIds = std::vector<EdgeId>;

    // Constructor
    NeuralNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& outputNodes);

    inline auto getNodes() const->NodeIds;

    inline bool hasNode(NodeId id) const { return m_nodes.find(id) != m_nodes.end(); }
    inline auto getNode(NodeId id) const->const Node&;
    inline void setNodeValue(NodeId id, float value);

    inline bool hasEdge(EdgeId id) const { return m_edges.find(id) != m_edges.end(); }
    inline float getWeight(EdgeId id) const;
    inline void setWeight(EdgeId id, float weight);
    inline auto getInNode(EdgeId id) const->NodeId;
    inline auto getOutNode(EdgeId id) const->NodeId;

    auto getOutputNodes() const->NodeIds;

    // Evaluates this network and calculate new values for each node.
    virtual void evaluate();

    // Returns false if this network has invalid data.
    virtual bool validate() const;

protected:

    // Data used evaluation
    struct EvaluationData
    {
        enum class NodeState
        {
            NONE,
            EVALUATED
        };

        EvaluationData(const NeuralNetwork* network);

        inline NodeState getNodeState(NodeId id) const { return m_nodeStates[m_id2Index[id]]; }
        inline void setNodeState(NodeId id, NodeState state) { m_nodeStates[m_id2Index[id]] = state; }

        std::unordered_map<NodeId, int> m_id2Index;
        std::vector<NodeState> m_nodeStates;
    };

    inline auto accessNode(NodeId id)->Node&;

    void evaluateNodeRecursive(NodeId id, EvaluationData& data);

    // Used by validate().
    bool hasCircularEdges() const;
    bool hasCircularEdgesRecursive(NodeId id, std::unordered_set<NodeId>& visitedNodes) const;

    Nodes m_nodes;
    Edges m_edges;
    NodeIds m_outputNodes;
};


template <typename Node, typename Edge>
NeuralNetwork<Node, Edge>::NeuralNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& outputNodes)
    : m_nodes(nodes)
    , m_edges(edges)
    , m_outputNodes(outputNodes)
{
    if (!validate())
    {
        WARN("Input nodes and edges are not valid neural network.");
    }
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getNodes() const->NodeIds
{
    NodeIds idsOut;
    idsOut.reserve(m_nodes.size());

    for (auto itr : m_nodes)
    {
        idsOut.push_back(itr->first);
    }

    return idsOut;
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getNode(NodeId id) const->const Node&
{
    assert(hasNode(id));
    return m_nodes[id];
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::accessNode(NodeId id)->Node&
{
    assert(hasNode(id));
    return m_nodes[id];
}

template <typename Node, typename Edge>
inline void NeuralNetwork<Node, Edge>::setNodeValue(NodeId id, float value)
{
    if (hasNode(id))
    {
        m_nodes[id].setValue(value);
    }
    else
    {
        WARN("Trying to set a value for a node which doesn't exist.");
    }
}

template <typename Node, typename Edge>
inline float NeuralNetwork<Node, Edge>::getWeight(EdgeId id) const
{
    assert(hasEdge(id));
    return m_edges[id].getWeight();
}

template <typename Node, typename Edge>
inline void NeuralNetwork<Node, Edge>::setWeight(EdgeId id, float weight)
{
    assert(hasEdge(id));
    m_edges[id].setWeight(weight);
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getInNode(EdgeId id) const->NodeId
{
    assert(hasEdge(id));
    return m_edges[id].getInNode();
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getOutNode(EdgeId id) const->NodeId
{
    assert(hasEdge(id));
    return m_edges[id].getOutNode();
}


template <typename Node, typename Edge>
auto NeuralNetwork<Node, Edge>::getOutputNodes() const->NodeIds
{
    NodeIds nodesOut;
    nodesOut.reserve(m_outputNodes.size());
    for (NodeId outputNode : m_outputNodes)
    {
        nodesOut.insert(outputNode);
    }

    return nodesOut;
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::evaluate()
{
    assert(validate());

    // Initialize evaluation data
    EvaluationData data(this);

    // Evaluate output nodes
    for (NodeId id : m_outputNodes)
    {
        evaluateNodeRecursive(id, data);
    }
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::evaluateNodeRecursive(NodeId id, EvaluationData& data)
{
    const Node& node = getNode(id);

    // Calculate value of this node by visiting all parent nodes.
    float sumValue = 0;
    for (EdgeId incomingId : node.getIncomingEdges())
    {
        NodeId inNodeId = getInNode(incomingId);

        // Recurse if we haven't evaluated this parent node yet.
        // NOTE: We assume that the network doesn't have any circular edges.
        if (data.getNodeState(inNodeId) != EvaluationData::NodeState::EVALUATED)
        {
            evaluateNodeRecursive(incomingId, data);
        }

        // Add a value from this parent.
        sumValue += getNode(inNodeId).getValue() * getWeight(incomingId);
    }

    data.setNodeState(id, EvaluationData::NodeState::EVALUATED);
    accessNode(id).setValue(sumValue);
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::validate() const
{
    if (m_nodes.size() < 2) return false;
    if (m_edges.size() < 1) return false;
    if (m_outputNodes.size() < 1) return false;

    // Validate all edges.
    for (auto itr : m_edges)
    {
        const Edge& e = itr->second;
        if (!hasNode(e.getInNode())) return false;
        if (!hasNode(e.getOutNode())) return false;
    }

    // Validate all output nodes.
    for (NodeId n : m_outputNodes)
    {
        if (!hasNode(n)) return false;

        // Make sure that no edge has this node as its inNode.
        for (const Edge& e : m_edges)
        {
            if (e.getInNode() == n) return false;
        }
    }

    // Validate all nodes.
    int numInputOrBiasNode = 0;
    for (auto itr : m_nodes)
    {
        const Node& n = itr->second;
        if (n.getIncomingEdges().size() == 0)
        {
            numInputOrBiasNode++;
            continue;
        }

        for (EdgeId e : n.getIncomingEdges())
        {
            if (!hasEdge(e)) return false;
        }
    }
    if (numInputOrBiasNode == 0) return false;

    // Make sure the the network doesn't contain circular edges.
    if (hasCircularEdges()) return false;

    return true;
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::hasCircularEdgesRecursive(NodeId id, std::unordered_set<NodeId>& visitedNodes) const
{
    if (visitedNodes.find(id) != visitedNodes.end())
    {
        // Already visited this node.
        return true;
    }

    visitedNodes.insert(id);

    const Node& n = getNode(id);
    for (EdgeId e : n.getIncomingEdges())
    {
        if (hasCircularEdgesRecursive(e.getInNode(), visitedNodes))
        {
            return true;
        }
    }

    return false;
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::hasCircularEdges() const
{
    std::unordered_set<NodeId> checkedNodes;

    for (auto itr : m_nodes)
    {
        NodeId id = itr->first;

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
}

template <typename Node, typename Edge>
NeuralNetwork<Node, Edge>::EvaluationData::EvaluationData(const NeuralNetwork* network)
{
    NodeIds nodeIds = network->getNodes();
    const int numNodes = nodeIds.size();

    m_id2Index.reserve(numNodes);
    int counter = 0;
    for (NodeId node : nodeIds)
    {
        m_id2Index.insert(node, counter++);
    }

    m_nodeStates.resize(numNodes, NodeState::NONE);
}
