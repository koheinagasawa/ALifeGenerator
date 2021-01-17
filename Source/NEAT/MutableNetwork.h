/*
* NeuralNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork.h>

// Edge which can be turned on and off without losing previous weight value.
struct SwitchableEdge : public EdgeBase
{
    SwitchableEdge(NodeId inNode, NodeId outNode, float weight = 1.f, bool enabled = true);
    SwitchableEdge();

    virtual NodeId getInNode() const override;
    virtual NodeId getOutNode() const override;
    virtual float getWeight() const override;
    virtual void setWeight(float weight) override;

    inline bool isEnabled() const { return m_enabled; }
    inline void setEnabled(bool enable) { m_enabled = enable; }

protected:
    NodeId m_inNode, m_outNode;
    float m_weight;
    bool m_enabled;
};

struct SwitchableEdge;

// Mutable network for NEAT.
template <typename Node>
class MutableNetwork : public NeuralNetwork<Node, SwitchableEdge>
{
public:
    // Declarations of types.
    using Edge = SwitchableEdge;
    using Base = NeuralNetwork<Node, Edge>;
    using Nodes = Base::Nodes;
    using Edges = Base::Edges;
    using NodeData = Base::NodeData;
    using NodeDatas = Base::NodeDatas;
    using NodeIds = Base::NodeIds;
    using EdgeIds = Base::EdgeIds;

    // Constructor using pre-setup network data.
    MutableNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& outputNodes);

    // Add a new node by dividing the edge at edgeId.
    // Ids of newly created node and edge will be set to newNodeIdOut and newEdgeIdOut.
    void addNodeAt(EdgeId edgeId, NodeId& newNodeIdOut, EdgeId& newEdgeIdOut);

    // Add a new edge between node1 and node2 with weight.
    // Returned value is edge id of the newly created edge.
    EdgeId addEdgeAt(NodeId node1, NodeId node2, float weight = 1.0f);

    // Enable/disable an edge.
    void setEdgeEnabled(EdgeId edgeId, bool enable);

    bool isEdgeEnabled(EdgeId edgeId) const;

protected:
    NodeId m_maxNodeId;
    EdgeId m_maxEdgeId;
};

template <typename Node>
MutableNetwork<Node>::MutableNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& outputNodes)
    : Base(nodes, edges, outputNodes)
    , m_maxNodeId(0)
    , m_maxEdgeId(0)
{
    for (const auto& itr : this->m_nodes)
    {
        if (m_maxNodeId < itr.first)
        {
            m_maxNodeId = itr.first;
        }
    }

    for (const auto& itr : this->m_edges)
    {
        if (m_maxEdgeId < itr.first)
        {
            m_maxEdgeId = itr.first;
        }
    }
}

template <typename Node>
void MutableNetwork<Node>::addNodeAt(EdgeId edgeId, NodeId& newNodeIdOut, EdgeId& newEdgeIdOut)
{
    // Make sure that edgeId exists.
    if (!this->hasEdge(edgeId))
    {
        WARN("Edge id doesn't exist.");
        newNodeIdOut = NodeId::invalid();
        newEdgeIdOut = EdgeId::invalid();
        return;
    }

    // Create a new node.
    NodeData newNode;
    {
        m_maxNodeId = m_maxNodeId.val() + 1;
        newNodeIdOut = m_maxNodeId;
        newNode.m_incomingEdges.push_back(edgeId);
        this->m_nodes[newNodeIdOut] = newNode;
    }

    // Update outNode of the divided edge
    const Edge& edgeToDivide = this->m_edges.at(edgeId);
    NodeId outNodeId = edgeToDivide.getOutNode();
    this->m_edges[edgeId] = Edge(edgeToDivide.getInNode(), newNodeIdOut, edgeToDivide.getWeight(), edgeToDivide.isEnabled());

    // Create a new edge between the new node and the original out node.
    Edge newEdge(newNodeIdOut, outNodeId, 1.0f);
    {
        m_maxEdgeId = m_maxEdgeId.val() + 1;
        newEdgeIdOut = m_maxEdgeId;
        this->m_edges[newEdgeIdOut] = newEdge;
    }

    // Update incoming edges of the out node.
    EdgeIds& edgesToOutNode = this->m_nodes[outNodeId].m_incomingEdges;
    for (auto itr = edgesToOutNode.begin(); itr != edgesToOutNode.end(); ++itr)
    {
        if (*itr == edgeId)
        {
            edgesToOutNode.erase(itr);
            break;
        }
    }
    edgesToOutNode.push_back(newEdgeIdOut);
}

template <typename Node>
auto MutableNetwork<Node>::addEdgeAt(NodeId node1, NodeId node2, float weight /* = 1.0f */)->EdgeId
{
    // Make sure that node1 and node2 exist.
    if(!this->hasNode(node1) || !this->hasNode(node2))
    {
        WARN("At least one of the give node ids doesn't exist.");
        return EdgeId::invalid();
    }

    NodeData& outNodeData = this->m_nodes[node2];
    EdgeIds& edgesToOutNode = outNodeData.m_incomingEdges;

    // Check if there is already an edge between the two nodes.
    for (EdgeId eid : edgesToOutNode)
    {
        if (this->getInNode(eid) == node1)
        {
            WARN("There is already an edge between the given two nodes.");
            return EdgeId::invalid();
        }
    }

    // Make sure that the node1 is not output node.
    for (NodeId id : this->m_outputNodes)
    {
        if (id == node1)
        {
            WARN("Output node cannot have an outgoing edge. Abort adding a new edge.");
            return EdgeId::invalid();
        }
    }

    // Create a new edge
    Edge newEdge(node1, node2, weight);
    EdgeId newEdgeId = m_maxEdgeId.val() + 1;
    this->m_edges[newEdgeId] = newEdge;

    edgesToOutNode.push_back(newEdgeId);

    // Make sure that the new edge doesn't cause circular networks
    if (this->hasCircularEdges())
    {
        // Revert the change
        this->m_edges.erase(newEdgeId);
        edgesToOutNode.pop_back();
        WARN("Cannot add an edge because it would cause a circular network.");
        return EdgeId::invalid();
    }

    m_maxEdgeId = m_maxEdgeId.val() + 1;
    return newEdgeId;
}

template <typename Node>
void MutableNetwork<Node>::setEdgeEnabled(EdgeId edgeId, bool enable)
{
    this->m_edges[edgeId].setEnabled(enable);
}

template <typename Node>
bool MutableNetwork<Node>::isEdgeEnabled(EdgeId edgeId) const
{
    return this->m_edges.at(edgeId).isEnabled();
}

