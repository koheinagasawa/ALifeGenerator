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

    // Copy and move constructor and operator
    SwitchableEdge(const SwitchableEdge& other) = default;
    SwitchableEdge(SwitchableEdge&& other) = default;
    void operator=(const SwitchableEdge& other);
    void operator=(SwitchableEdge&& other);

    virtual NodeId getInNode() const override;
    virtual NodeId getOutNode() const override;
    virtual float getWeight() const override;
    virtual void setWeight(float weight) override;

    inline bool isEnabled() const { return m_enabled; }
    inline void setEnabled(bool enable) { m_enabled = enable; }

    // Return weight regardless of whether this edge is enabled.
    inline float getWeightRaw() const { return m_weight; }

protected:
    const NodeId m_inNode, m_outNode;
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

    // Copy constructor
    MutableNetwork(const MutableNetwork& other) = default;

    // Add a new node by dividing the edge at edgeId.
    bool addNodeAt(EdgeId edgeId, NodeId newNodeId, EdgeId newIncomingEdgeId, EdgeId newOutgoingEdgeId);

    // Add a new edge between node1 and node2 with weight.
    bool addEdgeAt(NodeId node1, NodeId node2, EdgeId newEdgeId, float weight = 1.0f);

    // Replace an node id with a new node id
    void replaceNodeId(NodeId nodeId, NodeId newId);

    // Replace an edge id with a new edge id
    void replaceEdgeId(EdgeId edgeId, EdgeId newId);

    // Enable/disable an edge.
    void setEdgeEnabled(EdgeId edgeId, bool enable);
    bool isEdgeEnabled(EdgeId edgeId) const;

    // Return weight regardless of whether this edge is enabled.
    float getWeightRaw(EdgeId edgeId) const;
};

template <typename Node>
MutableNetwork<Node>::MutableNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& outputNodes)
    : Base(nodes, edges, outputNodes)
{
}

template <typename Node>
bool MutableNetwork<Node>::addNodeAt(EdgeId edgeId, NodeId newNodeId, EdgeId newIncomingEdgeId, EdgeId newOutgoingEdgeId)
{
    assert(this->validate());
    assert(!this->hasNode(newNodeId) && !this->hasEdge(newIncomingEdgeId) && !this->hasEdge(newOutgoingEdgeId));

    // Make sure that edgeId exists.
    if (!this->hasEdge(edgeId))
    {
        WARN("Edge id doesn't exist.");
        return false;
    }

    // Disable the divided edge
    const Edge& edgeToDivide = this->m_edges.at(edgeId);
    const float weight = edgeToDivide.getWeight();
    this->m_edges[edgeId].setEnabled(false);

    // Create two new edges.
    Edge newEdge1(edgeToDivide.getInNode(), newNodeId, 1.0f);
    {
        this->m_edges.insert({ newIncomingEdgeId, newEdge1 });
    }
    Edge newEdge2(newNodeId, edgeToDivide.getOutNode(), weight);
    {
        this->m_edges.insert({ newOutgoingEdgeId, newEdge2 });
    }

    // Create a new node.
    NodeData newNode;
    newNode.m_incomingEdges.push_back(newIncomingEdgeId);
    this->m_nodes.insert({ newNodeId, newNode });

    // Update incoming edges of the out node.
    this->m_nodes[edgeToDivide.getOutNode()].m_incomingEdges.push_back(newOutgoingEdgeId);

    assert(this->validate());

    return true;
}

template <typename Node>
bool MutableNetwork<Node>::addEdgeAt(NodeId node1, NodeId node2, EdgeId newEdgeId, float weight /* = 1.0f */)
{
    assert(this->validate());
    assert(!this->hasEdge(newEdgeId));

    // Make sure that node1 and node2 exist.
    if(!this->hasNode(node1) || !this->hasNode(node2))
    {
        WARN("At least one of the give node ids doesn't exist.");
        return false;
    }

    NodeData& outNodeData = this->m_nodes[node2];
    EdgeIds& edgesToOutNode = outNodeData.m_incomingEdges;

    // Check if there is already an edge between the two nodes.
    for (EdgeId eid : edgesToOutNode)
    {
        if (this->getInNode(eid) == node1)
        {
            WARN("There is already an edge between the given two nodes.");
            return false;
        }
    }

    // Make sure that the node1 is not output node.
    for (NodeId id : this->m_outputNodes)
    {
        if (id == node1)
        {
            WARN("Output node cannot have an outgoing edge. Abort adding a new edge.");
            return false;
        }
    }

    // Create a new edge
    this->m_edges.insert({ newEdgeId, Edge(node1, node2, weight) });

    edgesToOutNode.push_back(newEdgeId);

    // Make sure that the new edge doesn't cause circular networks
    if (this->hasCircularEdges())
    {
        // Revert the change
        this->m_edges.erase(newEdgeId);
        edgesToOutNode.pop_back();
        WARN("Cannot add an edge because it would cause a circular network.");
        return false;
    }

    assert(this->validate());

    return true;
}

template <typename Node>
void MutableNetwork<Node>::replaceNodeId(NodeId nodeId, NodeId newId)
{
    assert(this->validate());
    assert(this->hasNode(nodeId));
    assert(!this->hasNode(newId));

    // Replace nodeIds stored in edges.
    for (auto& itr : this->m_edges)
    {
        Edge& edge = itr.second;
        if (edge.getInNode() == nodeId)
        {
            itr.second = Edge(newId, edge.getOutNode(), edge.getWeight(), edge.isEnabled());
        }
        else if (edge.getOutNode() == nodeId)
        {
            itr.second = Edge(edge.getInNode(), newId, edge.getWeight(), edge.isEnabled());
        }
    }

    // Replace the node itself
    const NodeData& nd = this->m_nodes.at(nodeId);
    this->m_nodes.insert({ newId, nd });
    this->m_nodes.erase(nodeId);

    for (auto itr = this->m_outputNodes.begin(); itr != this->m_outputNodes.end(); itr++)
    {
        if (*itr == nodeId)
        {
            this->m_outputNodes.erase(itr);
            this->m_outputNodes.push_back(newId);
            break;
        }
    }

    assert(this->validate());
}

template <typename Node>
void MutableNetwork<Node>::replaceEdgeId(EdgeId edgeId, EdgeId newId)
{
    assert(this->validate());
    assert(this->hasEdge(edgeId));
    assert(!this->hasEdge(newId));

    const Edge& edge = this->m_edges.at(edgeId);

    // Update incoming edges of output node
    NodeData& outputNode = this->m_nodes[edge.getOutNode()];
    EdgeIds& edgesToOutNode = outputNode.m_incomingEdges;
    for (EdgeId& e : edgesToOutNode)
    {
        if (e == edgeId)
        {
            e = newId;
            break;
        }
    }

    // Replace the edge.
    const Edge e = this->getEdges().at(edgeId);

    // Remove the original edge id
    for (auto itr = this->m_edges.begin(); itr != this->m_edges.end(); itr++)
    {
        if (itr->first == edgeId)
        {
            this->m_edges.erase(itr);
            break;
        }
    }

    // Add the new edge id
    this->m_edges.insert({ newId, e });

    assert(this->validate());
}

template <typename Node>
void MutableNetwork<Node>::setEdgeEnabled(EdgeId edgeId, bool enable)
{
    assert(this->validate());

    this->m_edges[edgeId].setEnabled(enable);

#ifdef _DEBUG
    if (enable && this->hasCircularEdges())
    {
        WARN("Cannot enabling edge %d because it would make this network circular.", edgeId.val());
        this->m_edges[edgeId].setEnabled(false);
    }
#endif

    assert(this->validate());
}

template <typename Node>
bool MutableNetwork<Node>::isEdgeEnabled(EdgeId edgeId) const
{
    return this->m_edges.at(edgeId).isEnabled();
}

template <typename Node>
float MutableNetwork<Node>::getWeightRaw(EdgeId edgeId) const
{
    return this->m_edges.at(edgeId).getWeightRaw();
}

