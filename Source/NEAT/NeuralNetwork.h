/*
* NeuralNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <vector>
#include <unordered_map>
#include <stack>

#include <Common/BaseType.h>

template <typename Node, typename Edge>
class NeuralNetwork
{
public:

    // Declarations of types.
    DECLARE_ID(NodeId);
    DECLARE_ID(EdgeId);

    using Nodes = std::unordered_map<NodeId, Node>;
    using Edges = std::unordered_map<EdgeId, Edge>;
    using NodeIds = std::vector<NodeId>;
    using EdgeIds = std::vector<EdgeId>;

    // Base struct of node.
    struct NodeBase
    {
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
    void evaluate();

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

    for (Nodes::iterator itr : m_nodes)
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
        evaluateNodeRecursive(id, data, nodes);
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
NeuralNetwork<Node, Edge>::EvaluationData::EvaluationData(const NeuralNetwork* network)
{
    NodeIds nodeIds = network->getNodes();
    const int numNodes = nodes.size();

    m_id2Index.reserve(numNodes);
    int counter = 0;
    for (NodeId node : nodeIds)
    {
        m_id2Index.insert(node, counter++);
    }

    m_nodeStates.resize(numNodes, NONE);
}
