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
    using NodeEntry = std::pair<NodeId, NodeData>;
    using EdgeEntry = std::pair<EdgeId, Edge>;

    // Constructor from network information
    NeuralNetwork(const Nodes& nodes, const Edges& edges);

    // Copy constructor
    NeuralNetwork(const NeuralNetwork& other) = default;

    inline int getNumNodes() const { return (int)m_nodes.size(); }
    inline auto getNodes() const->const NodeDatas& { return m_nodes; }

    inline bool hasNode(NodeId id) const { return m_nodes.find(id) != m_nodes.end(); }
    inline auto getNode(NodeId id) const->const Node&;
    inline auto accessNode(NodeId id)->Node&;
    inline void setNodeValue(NodeId id, float value);
    inline auto getIncomingEdges(NodeId id) const->EdgeIds;
    inline bool isConnected(NodeId node1, NodeId node2) const;

    inline int getNumEdges() const { return (int)m_edges.size(); }
    inline auto getEdges() const->const Edges& { return m_edges; }
    inline bool hasEdge(EdgeId id) const { return m_edges.find(id) != m_edges.end(); }
    inline float getWeight(EdgeId id) const;
    inline void setWeight(EdgeId id, float weight);
    inline auto getInNode(EdgeId id) const->NodeId;
    inline auto getOutNode(EdgeId id) const->NodeId;

    // Returns false if this network has invalid data.
    virtual bool validate() const;

protected:
    // Default constructor.
    NeuralNetwork() = default;

    // Construct m_nodes. This is called from constructor.
    void constructNodeData(const Nodes& nodes);

    NodeDatas m_nodes; // Nodes of this network.
    Edges m_edges; // Edges of this network.

    // [TODO] Support recurrent network. Or at least make the entire system flexible so that it works with
    //        recurrent network when we support it in the future.
};

template <typename Node, typename Edge>
NeuralNetwork<Node, Edge>::NeuralNetwork(const Nodes& nodes, const Edges& edges)
    : m_edges(edges)
{
    constructNodeData(nodes);
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::constructNodeData(const Nodes& nodes)
{
    // Allocate NodeDatas
    m_nodes.clear();
    m_nodes.reserve(nodes.size());
    for (const auto& itr : nodes)
    {
        m_nodes.insert({ itr.first, NodeData{ itr.second } });
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
inline bool NeuralNetwork<Node, Edge>::isConnected(NodeId node1, NodeId node2) const
{
    assert(hasNode(node1) && hasNode(node2) && node1 != node2);

    for (EdgeId e : m_nodes.at(node1).m_incomingEdges)
    {
        if (getInNode(e) == node2)
        {
            return true;
        }
    }

    for (EdgeId e : m_nodes.at(node2).m_incomingEdges)
    {
        if (getInNode(e) == node1)
        {
            return true;
        }
    }

    return false;
}

template <typename Node, typename Edge>
inline float NeuralNetwork<Node, Edge>::getWeight(EdgeId id) const
{
    if (!hasEdge(id))
    {
        int i = 0;
        i = 1;
    }
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
bool NeuralNetwork<Node, Edge>::validate() const
{
#ifdef DEBUG_SLOW
    if (m_nodes.size() < 2) return false;
    if (m_edges.size() < 1) return false;

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
#endif
    return true;
}
