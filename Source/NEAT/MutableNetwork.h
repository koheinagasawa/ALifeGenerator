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
    using Base = NeuralNetwork<Node, SwitchableEdge>;
    using Nodes = Base::Nodes;
    using Edges = Base::Edges;
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

    // Get innovation id of an edge.
    auto getInnovationId(EdgeId edgeId) const->InnovationId;

    // Get innovations of this network. Returned list of innovation entries is sorted by innovation id.
    auto getInnovations() const->const InnovationEntries&;

protected:
    InnovationEntries m_innovations; // A list of innovations sorted by innovation id.
};
