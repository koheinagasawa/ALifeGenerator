/*
* GenomeBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <memory>
#include <functional>

#include <EvoAlgo/NeuralNetwork/NeuralNetwork.h>
#include <EvoAlgo/NeuralNetwork/BakedNeuralNetwork.h>

class NeuralNetworkEvaluator;

// Base class of genome used for genetic algorithms.
class GenomeBase
{
public:
    // Type declarations.
    using Node = DefaultNode;
    using Edge = DefaultEdge;
    using Network = NeuralNetwork<Node, Edge>;
    using NetworkPtr = std::shared_ptr<Network>;
    using BakedNetworkPtr = std::shared_ptr<BakedNeuralNetwork>;

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
    inline auto accessNetwork() const->NetworkPtr { return m_network; }

    //
    // Edge interface
    //

    // Get weight of edge.
    inline float getEdgeWeight(EdgeId edgeId) const { return m_network->getWeight(edgeId); }

    // Set weight of edge.
    inline void setEdgeWeight(EdgeId edgeId, float weight) { m_network->setWeight(edgeId, weight); m_needRebake = true; }

    // Get weight of edge regardless if it's enabled or not.
    inline float getEdgeWeightRaw(EdgeId edgeId) const { return m_network->getEdge(edgeId).getWeightRaw(); }

    // Return true if the edge is enabled.
    inline bool isEdgeEnabled(EdgeId edgeId) const { return m_network->getEdge(edgeId).isEnabled(); }

    // Set enable/disable the edge.
    inline void setEdgeEnabled(EdgeId edgeId, bool enabled) { m_network->accessEdge(edgeId).setEnabled(enabled); m_needRebake = true; }

    // Return the number of edges.
    inline int getNumEdges() const { return m_network->getNumEdges(); }

    // Return the total number of enabled edges.
    int getNumEnabledEdges() const;

    //
    // Node interface
    //

    // Clear all values stored in nodes to zero.
    void clearNodeValues();

    // Set values of input nodes.
    // values has to be the same size as the number of input nodes (m_inputNodes) and has to be sorted in the same order as them.
    void setInputNodeValues(const std::vector<float>& values, float biasNodeValue = 0.f);

    // Set value of bias node.
    void setBiasNodeValue(float value);

    // Get node id of the bias node.
    inline NodeId getBiasNode() const { return m_biasNode; }

    // Get a value of the node.
    float getNodeValue(NodeId nodeId) const;

    // Get input nodes of the network.
    inline auto getInputNodes() const->const Network::NodeIds& { return m_network->getInputNodes(); }

    // Get output nodes of the network.
    inline auto getOutputNodes() const->const Network::NodeIds& { return m_network->getOutputNodes(); }

    // Return the number of nodes.
    inline int getNumNodes() const { return m_network->getNumNodes(); }

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

    // Evaluate this genome using the current values of input nodes.
    void evaluate();

    // Evaluate this genome using the current values of input nodes and the provided evaluator.
    void evaluate(NeuralNetworkEvaluator* evaluator);

protected:
    // Bake the newtork.
    void bake();

    // Return ture if baked network should also update its node values.
    inline bool shouldUpdateBakedNetworkNode() const { return m_bakedNetwork && !m_needRebake; }

    NetworkPtr m_network;                   // The network.
    BakedNetworkPtr m_bakedNetwork;         // The baked network for faster evaluation.
    NodeId m_biasNode = NodeId::invalid();  // The bias node.
    bool m_needRebake = true;               // True if network has any structural changes and rebake is required.
};
