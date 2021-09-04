/*
* BakedNeuralNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <vector>
#include <unordered_map>
#include <NEAT/NeuralNetwork/Node.h>
#include <NEAT/NeuralNetwork/Edge.h>

template<typename Node, typename Edge>
class NeuralNetwork;

// Neural network whose structure is fixed but much faster to evaluate than general NeuralNetwork type.
class BakedNeuralNetwork
{
public:
    // Type definition
    using ActivationFunc = const std::function<float(float)>*;
    using Network = NeuralNetwork<DefaultNode, DefaultEdge>;
    using NodeIdIndexMap = std::unordered_map<NodeId, unsigned int>;

    // Constructors
    BakedNeuralNetwork(const Network* network);
    BakedNeuralNetwork(const BakedNeuralNetwork& other) = default;

    // Set value of a node.
    void setNodeValue(NodeId node, float value);

    // Get value of a node.
    float getNodeValue(NodeId node) const;

    // Set all node values to zero.
    void clearNodeValues();

    // Evaluate this network.
    void evaluate();

    // Return true if this network contains circular connections.
    inline bool isCircularNetwork() const { return m_isCircularNetwork; }

private:

    // Node data
    struct Node
    {
        int m_startEdge;                    // Index of m_edges where edges for this node starts.
        unsigned short m_numEdges;          // The number of edges coming to this node.
        unsigned short m_activationFunc;    // Index of activation function.
        float m_value;                      // Value of this node. We need to store both raw value and activated value for recursive network.
        float m_activatedValue;             // Activated value of this node.
    };

    struct Edge
    {
        unsigned int m_node;                // In node of this edge.
        float m_weight;                     // The weight.
    };

    std::vector<Node> m_nodes;                      // List of nodes. They are sorted so that they can evaluate from the first node to the end without revisiting previous nodes.
    std::vector<Edge> m_edges;                      // List of edges in the order of their out nodes.
    std::vector<ActivationFunc> m_activationFuncs;  // List of activation functions.
    NodeIdIndexMap m_nodeIdIndexMap;                // Map from NodeId to index of m_nodes.
    const bool m_isCircularNetwork;                 // True if this network has any circular connections.

    static const std::function<float(float)> s_nullActivation;  // Null activation (activatedValue == value)
};
