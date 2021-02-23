/*
* GenomeBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <memory>
#include <functional>

#include <NEAT/MutableNetwork.h>

// Base class of genome used for genetic algorithms.
class GenomeBase
{
public:
    // Wrapper struct for activation function.
    struct Activation
    {
        using Func = std::function<float(float)>;

        Activation(const Func& func) : m_func(func) {}

        float activate(float value) const { return m_func(value); }

        std::string m_name;
        const Func m_func;
    };

    // Node structure of the genome.
    struct Node : public NodeBase
    {
    public:
        // Type of Node.
        enum class Type
        {
            INPUT,
            HIDDEN,
            OUTPUT,
            NONE
        };

        // Default constructor. This is used only by container of Node in Network class and users shouldn't call it.
        // Use Node(Type type) instead.
        Node() = default;

        // Constructor with node type.
        Node(Type type);

        // Copy constructor.
        Node(const Node& other) = default;

        virtual float getValue() const override;
        virtual void setValue(float value) override;

        inline void setActivation(const Activation* activation) { m_activation = activation; }
        inline auto getActivationName() const->const std::string& { return m_activation->m_name; }

        inline Type getNodeType() const { return m_type; }

    protected:
        float m_value = 0.f;
        Type m_type = Type::NONE;
        const Activation* m_activation = nullptr;

        friend class GenomeBase;
    };

    // Type declarations.
    using Network = MutableNetwork<Node>;
    using NetworkPtr = std::shared_ptr<Network>;

    // Constructor
    GenomeBase(const Activation* defaultActivation);

    // Copy constructor and operator.
    GenomeBase(const GenomeBase& other);
    void operator= (const GenomeBase& other);

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

    //
    // Node interface
    //

    inline auto getInputNodes() const->const Network::NodeIds& { return m_inputNodes; }

    // Set values of input nodes.
    // values has to be the same size as the number of input nodes (m_inputNodes) and has to be sorted in the same order as them.
    void setInputNodeValues(const std::vector<float>& values) const;

    //
    // Activation interface
    //

    // Set activation of node.
    inline void setActivation(NodeId nodeId, const Activation* activation) { m_network->accessNode(nodeId).m_activation = activation; }

    // Set activation of all nodes except input nodes.
    void setActivationAll(const Activation* activation);

    // Get the default activation.
    inline auto getDefaultActivation() const->const Activation* { return m_defaultActivation; }

    // Set the default activation.
    inline void setDefaultActivation(const Activation* activation) { m_defaultActivation = activation; }

    //
    // Evaluation
    //

    // Evaluate this genome using the given input nodes.
    // inputNodeValues has to be the same size as the number of input nodes (m_inputNodes) and has to be sorted in the same order as them.
    void evaluate(const std::vector<float>& inputNodeValues) const;

    // Evaluate this genome using the current values of input nodes.
    void evaluate() const;

protected:
    void setNodeTypeAndActivation(NodeId node, Node::Type type, const Activation* activation);

    NetworkPtr m_network;                   // The network.
    Network::NodeIds m_inputNodes;          // A list of input nodes.
    const Activation* m_defaultActivation;  // Activation assigned to new Node by default.
};
