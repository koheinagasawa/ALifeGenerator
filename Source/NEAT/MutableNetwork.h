/*
* NeuralNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork.h>

DECLARE_ID(InnovationId);

// Singleton class to control InnovationId.
class InnovationCounter
{
public:
    static InnovationId getNewInnovationId();

    static void reset();

protected:
    InnovationCounter() = default;
    InnovationCounter(const InnovationCounter&) = delete;
    void operator=(const InnovationCounter&) = delete;

    static InnovationId s_id;
    static InnovationCounter s_instance;
};

// Edge which can be turned on and off without losing previous weight value.
struct SwitchableEdge : public EdgeBase
{
    SwitchableEdge(NodeId inNode, NodeId outNode, float weight = 1.f);

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

    struct InnovationEntry
    {
        InnovationId m_id;
        EdgeId m_edgeId;
    };

    struct Cinfo
    {
        uint16_t m_numInputNode;
        uint16_t m_numOutputNode;
    };

    using InnovationEntries = std::vector<InnovationEntry>;

    // Constructor with cinfo. It will construct the minimum dimensional network where there is no hidden node and
    // all input nodes and output nodes are fully connected.
    MutableNetwork(const Cinfo& cinfo);

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

    // Get innovations of this network. Returned list of innovation entries is sorted by innovation id.
    inline auto getInnovations() const->const InnovationEntries& { return m_innovations; }

protected:
    InnovationEntries m_innovations; // A list of innovations sorted by innovation id.
    NodeId m_maxNodeId;
    EdgeId m_maxEdgeId;
};

template <typename Node>
MutableNetwork<Node>::MutableNetwork(const Cinfo& cinfo)
{
    assert(cinfo.m_numInputNode > 0 && cinfo.m_numOutputNode > 0);

    const int numNodes = cinfo.m_numInputNode + cinfo.m_numOutputNode;
    NodeDatas& nodes = this->m_nodes;
    nodes.reserve(numNodes);

    int currentNodeId = 0;
    int currentEdgeId = 0;

    // Create input nodes.
    for (int i = 0; i < cinfo.m_numInputNode; i++)
    {
        nodes[NodeId(currentNodeId++)] = NodeData();
    }

    // Create output nodes and edges fully connected between input nodes and output nodes.
    NodeIds& outputNodes = this->m_outputNodes;
    outputNodes.reserve(cinfo.m_numOutputNode);
    Edges& edges = this->m_edges;
    for (int i = 0; i < cinfo.m_numOutputNode; i++)
    {
        NodeId id(currentNodeId++);
        NodeData outNodeData;
        outputNodes.push_back(id);

        // Create edges connected to this output node.
        outNodeData.m_incomingEdges.reserve(cinfo.m_numInputNode);
        for (int j = 0; j < cinfo.m_numInputNode; j++)
        {
            EdgeId eid(currentEdgeId++);
            Edge e(NodeId(j), id);
            edges[eid] = e;
            outNodeData.m_incomingEdges.push_back(eid);
        }

        nodes[id] = outNodeData;
    }

    m_maxNodeId = nodes.size() - 1;
    m_maxEdgeId = edges.size() - 1;
}

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
            m_maxEdgeId < itr.first;
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
        newNodeIdOut = m_maxNodeId;
        m_maxNodeId = m_maxNodeId + 1;
        newNode.m_incomingEdges.push_back(edgeId);
        this->m_nodes[newNodeIdOut] = newNode;
    }

    // Update outNode of the divided edge
    Edge& edgeToDivide = this->accessEdge(edgeId);
    NodeId outNodeId = edgeToDivide.getOutNode();
    edgeToDivide.m_outNode = newNodeIdOut;

    // Create a new edge between the new node and the original out node.
    Edge newEdge(newNodeIdOut, outNodeId, 1.0f);
    {
        newEdgeIdOut = m_maxEdgeId;
        m_maxEdgeId = m_maxEdgeId + 1;
        this->m_edges[newEdgeIdOut] = newEdge;
    }

    // Update incoming edges of the out node.
    EdgeIds& edgesToOutNode = this->m_nodes.at(outNodeId).m_incomingEdges;
    for (EdgeIds::iterator itr : edgesToOutNode)
    {
        if (*itr == edgeId)
        {
            edgesToOutNode.erase(itr);
            break;
        }
    }
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

    // Create a new edge
    Edge newEdge(node1, node2, weight);
    EdgeId newEdgeId = m_maxEdgeId;
    this->m_edges[newEdgeId] = newEdge;

    edgesToOutNode.push_back(newEdgeId);

    // Make sure that the new edge doesn't cause circular networks
    if (this->hasCircularEdges())
    {
        // Revert the change
        this->m_edges.erase(newEdgeId);
        edgesToOutNode.pop_back();
        return EdgeId::invalid();
    }

    m_maxEdgeId = m_maxEdgeId + 1;
    return newEdgeId;
}

template <typename Node>
void MutableNetwork<Node>::setEdgeEnabled(EdgeId edgeId, bool enable)
{
    this->accessEdge(edgeId).setEnabled(enable);
}
