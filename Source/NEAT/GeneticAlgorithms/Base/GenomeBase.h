/*
* GenomeBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <memory>
#include <functional>

#include <NEAT/NeuralNetwork/NeuralNetwork.h>

// Base class of genome used for genetic algorithms.
class GenomeBase
{
public:
    // Type declarations.
    using Node = DefaultNode;
    using Network = NeuralNetwork<Node, SwitchableEdge>;
    using NetworkPtr = std::shared_ptr<Network>;
    using Edge = SwitchableEdge;

    // Constructor
    GenomeBase() = default;

    // Copy constructor and operator.
    GenomeBase(const GenomeBase& other);
    void operator= (const GenomeBase& other);

    // Create a clone of this genome.
    virtual std::shared_ptr<GenomeBase> clone() const;

    //
    // Network
    //

    inline auto getNetwork() const->const Network* { return m_network.get(); }
    inline auto accessNetwork()->NetworkPtr { return m_network; }

    //
    // Edge interface
    //

    // Get weight of edge.
    inline float getEdgeWeight(EdgeId edgeId) const { return m_network->getWeight(edgeId); }

    // Set weight of edge.
    inline void setEdgeWeight(EdgeId edgeId, float weight) { m_network->setWeight(edgeId, weight); }

    // Get weight of edge regardless if it's enabled or not.
    inline float getEdgeWeightRaw(EdgeId edgeId) const { return m_network->getEdge(edgeId).getWeightRaw(); }

    // Return true if the edge is enabled.
    inline bool isEdgeEnabled(EdgeId edgeId) const { return m_network->getEdge(edgeId).isEnabled(); }

    // Set enable/disable the edge.
    inline void setEdgeEnabled(EdgeId edgeId, bool enabled) { m_network->accessEdge(edgeId).setEnabled(enabled); }

    // Return the total number of enabled edges.
    int getNumEnabledEdges() const;

    //
    // Node interface
    //

    // Clear all values stored in nodes to zero.
    void clearNodeValues() const;

    // Set values of input nodes.
    // values has to be the same size as the number of input nodes (m_inputNodes) and has to be sorted in the same order as them.
    void setInputNodeValues(const std::vector<float>& values, float biasNodeValue = 0.f) const;

    // Set value of bias node.
    void setBiasNodeValue(float value) const;

    // Get node id of the bias node.
    inline NodeId getBiasNode() const { return m_biasNode; }

    //
    // Activation interface
    //

    // Set activation of node.
    void setActivation(NodeId nodeId, const Activation* activation);

    // Set activation of all nodes except input nodes.
    void setActivationAll(const Activation* activation);

    //
    // Evaluation
    //

    // Evaluate this genome using the given input nodes.
    // inputNodeValues has to be the same size as the number of input nodes (m_inputNodes) and has to be sorted in the same order as them.
    void evaluate(const std::vector<float>& inputNodeValues, float biasNodeValue = 0.f) const;

    // Evaluate this genome using the current values of input nodes.
    void evaluate() const;

protected:
    NetworkPtr m_network;                   // The network.
    NodeId m_biasNode;                      // The bias node.
};
