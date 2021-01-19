/*
* NeuralNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <cassert>
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
    virtual float getValue() const = 0;
    virtual void setValue(float value) = 0;
};

// Base struct of edge.
struct EdgeBase
{
    virtual NodeId getInNode() const = 0;
    virtual NodeId getOutNode() const = 0;
    virtual float getWeight() const = 0;
    virtual void setWeight(float weight) = 0;
};

// Base class of neural network
template <typename Node, typename Edge>
class NeuralNetwork
{
public:
    // Declarations of types.
    using Nodes = std::unordered_map<NodeId, Node>;
    using Edges = std::unordered_map<EdgeId, Edge>;
    using NodeIds = std::vector<NodeId>;
    using EdgeIds = std::vector<EdgeId>;

    // Node and some additional data for shortcut access.
    struct NodeData
    {
        Node m_node;
        EdgeIds m_incomingEdges;
    };

    using NodeDatas = std::unordered_map<NodeId, NodeData>;

    // Constructor
    NeuralNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& outputNodes);

    inline auto getNodes() const->NodeIds;

    inline bool hasNode(NodeId id) const { return m_nodes.find(id) != m_nodes.end(); }
    inline auto getNode(NodeId id) const->const Node&;
    inline void setNodeValue(NodeId id, float value);
    inline auto getIncomingEdges(NodeId id) const->EdgeIds;

    inline int getNumEdges() const { return m_edges.size(); }
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
    // Default constructor.
    NeuralNetwork() = default;

    // Construct m_nodes. This is called from constructor.
    void constructNodeData(const Nodes& nodes);

    inline auto accessNode(NodeId id)->Node&;

    // Data used evaluation
    struct EvaluationData
    {
        enum class NodeState
        {
            NONE,
            EVALUATED
        };

        EvaluationData(const NeuralNetwork* network);

        inline NodeState getNodeState(NodeId id) const { return m_nodeStates[m_id2Index.at(id)]; }
        inline void setNodeState(NodeId id, NodeState state) { m_nodeStates[m_id2Index.at(id)] = state; }

        std::unordered_map<NodeId, int> m_id2Index; // Map between NodeId and its index in m_nodeStates.
        std::vector<NodeState> m_nodeStates; // Status of each node.
    };

    void evaluateNodeRecursive(NodeId id, EvaluationData& data);

    // Used by validate().
    bool hasCircularEdges() const;
    bool hasCircularEdgesRecursive(NodeId id, std::unordered_set<NodeId> visitedNodes) const;

    NodeDatas m_nodes; // Nodes of this network.
    Edges m_edges; // Edges of this network.
    NodeIds m_outputNodes; // A list of output nodes of this network.
};


template <typename Node, typename Edge>
NeuralNetwork<Node, Edge>::NeuralNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& outputNodes)
    : m_edges(edges)
    , m_outputNodes(outputNodes)
{
    constructNodeData(nodes);

    if (!validate())
    {
        WARN("Input nodes and edges are not valid neural network.");
    }
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::constructNodeData(const Nodes& nodes)
{
    // Allocate NodeDatas
    m_nodes.clear();
    m_nodes.reserve(nodes.size());
    for (const auto& itr : nodes)
    {
        m_nodes[itr.first] = NodeData{ itr.second };
    }

    // Set incoming edges array in each node
    for (const auto& itr : m_edges)
    {
        const Edge& e = itr.second;
        NodeId outNode = e.getOutNode();
        if (!hasNode(outNode))
        {
            WARN("Input edge contains invalid outNode value.");
            continue;
        }

        m_nodes[outNode].m_incomingEdges.push_back(itr.first);
    }
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getNodes() const->NodeIds
{
    NodeIds idsOut;
    idsOut.reserve(m_nodes.size());

    for (auto itr : m_nodes)
    {
        idsOut.push_back(itr.first);
    }

    return idsOut;
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getNode(NodeId id) const->const Node&
{
    assert(hasNode(id));
    return m_nodes.at(id).m_node;
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::accessNode(NodeId id)->Node&
{
    assert(hasNode(id));
    return m_nodes[id].m_node;
}

template <typename Node, typename Edge>
inline void NeuralNetwork<Node, Edge>::setNodeValue(NodeId id, float value)
{
    if (hasNode(id))
    {
        m_nodes[id].m_node.setValue(value);
    }
    else
    {
        WARN("Trying to set a value for a node which doesn't exist.");
    }
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getIncomingEdges(NodeId id) const->EdgeIds
{
    return m_nodes.at(id).m_incomingEdges;
}

template <typename Node, typename Edge>
inline float NeuralNetwork<Node, Edge>::getWeight(EdgeId id) const
{
    assert(hasEdge(id));
    return m_edges.at(id).getWeight();
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
    const Edge& e = m_edges.at(id);
    return e.getInNode();
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getOutNode(EdgeId id) const->NodeId
{
    assert(hasEdge(id));
    return m_edges.at(id).getOutNode();
}


template <typename Node, typename Edge>
auto NeuralNetwork<Node, Edge>::getOutputNodes() const->NodeIds
{
    NodeIds nodesOut;
    nodesOut.reserve(m_outputNodes.size());
    for (NodeId outputNode : m_outputNodes)
    {
        nodesOut.push_back(outputNode);
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
    NodeData& node = m_nodes[id];

    if (node.m_incomingEdges.empty())
    {
        // This is input/bias node. Don't update the value.
        return;
    }

    // Calculate value of this node by visiting all parent nodes.
    float sumValue = 0;
    for (EdgeId incomingId : node.m_incomingEdges)
    {
        NodeId inNodeId = getInNode(incomingId);

        // Recurse if we haven't evaluated this parent node yet.
        // NOTE: We assume that the network doesn't have any circular edges.
        if (data.getNodeState(inNodeId) != EvaluationData::NodeState::EVALUATED)
        {
            evaluateNodeRecursive(inNodeId, data);
        }

        // Add a value from this parent.
        sumValue += getNode(inNodeId).getValue() * getWeight(incomingId);
    }

    data.setNodeState(id, EvaluationData::NodeState::EVALUATED);
    node.m_node.setValue(sumValue);
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::validate() const
{
    if (m_nodes.size() < 2) return false;
    if (m_edges.size() < 1) return false;
    if (m_outputNodes.size() < 1) return false;

    // Validate all edges.
    {
        std::unordered_set<EdgeId> edges;
        for (const auto& itr : m_edges)
        {
            // Make sure that the id is unique.
            EdgeId id = itr.first;
            if (edges.find(id) != edges.end()) return false;
            edges.insert(id);

            const Edge& e = itr.second;
            if (!hasNode(e.getInNode())) return false;
            if (!hasNode(e.getOutNode())) return false;
        }
    }

    // Validate all output nodes.
    {
        std::unordered_set<NodeId> nodes;
        for (NodeId n : m_outputNodes)
        {
            if (!hasNode(n)) return false;

            // Make sure that the id is unique.
            if (nodes.find(n) != nodes.end()) return false;
            nodes.insert(n);

            if (getIncomingEdges(n).empty()) return false;

            // Make sure that no edge has this node as its inNode.
            for (const auto& itr : m_edges)
            {
                const Edge& e = itr.second;
                if (e.getInNode() == n) return false;
            }
        }
    }

    // Validate all nodes.
    {
        int numInputOrBiasNode = 0;
        std::unordered_set<NodeId> nodes;
        for (const auto& itr : m_nodes)
        {
            // Make sure that the id is unique.
            NodeId id = itr.first;
            if (nodes.find(id) != nodes.end()) return false;
            nodes.insert(id);

            if (getIncomingEdges(id).size() == 0)
            {
                numInputOrBiasNode++;
                continue;
            }

            std::unordered_set<EdgeId> edges;
            for (EdgeId e : getIncomingEdges(id))
            {
                if (!hasEdge(e)) return false;

                // Make sure that the id is unique.
                if (edges.find(e) != edges.end()) return false;
                edges.insert(e);
            }
        }
        if (numInputOrBiasNode == 0) return false;
    }


    // Make sure the the network doesn't contain circular edges.
    if (hasCircularEdges()) return false;

    return true;
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::hasCircularEdgesRecursive(NodeId id, std::unordered_set<NodeId> visitedNodes) const
{
    if (visitedNodes.find(id) != visitedNodes.end())
    {
        // Already visited this node.
        return true;
    }

    visitedNodes.insert(id);

    for (EdgeId e : getIncomingEdges(id))
    {
        if (hasCircularEdgesRecursive(m_edges.at(e).getInNode(), visitedNodes))
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

    for (const auto& itr : m_nodes)
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
NeuralNetwork<Node, Edge>::EvaluationData::EvaluationData(const NeuralNetwork* network)
{
    NodeIds nodeIds = network->getNodes();
    const int numNodes = nodeIds.size();

    m_id2Index.reserve(numNodes);
    int counter = 0;
    for (NodeId node : nodeIds)
    {
        m_id2Index[node] = counter++;
    }

    m_nodeStates.resize(numNodes, NodeState::NONE);
}
