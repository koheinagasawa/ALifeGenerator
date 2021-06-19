/*
* Node.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/Activation.h>

DECLARE_ID(NodeId);

// Base struct of node.
struct NodeBase
{
    virtual float getValue() const = 0;
    virtual void setValue(float value) = 0;
};

// Default node structure.
struct DefaultNode : public NodeBase
{
public:
    // Type of Node.
    enum class Type : char
    {
        INPUT,
        HIDDEN,
        OUTPUT,
        BIAS,
        NONE
    };

    // Default constructor.
    DefaultNode() = default;

    // Constructor with node type.
    DefaultNode(Type type);

    // Copy constructor.
    DefaultNode(const DefaultNode& other) = default;

    virtual float getValue() const override;
    virtual void setValue(float value) override;

    inline void setActivation(const Activation* activation) { m_activation = activation; }
    inline auto getActivationName() const->const std::string& { return m_activation->m_name; }

    inline void setNodeType(Type type) { m_type = type; }
    inline Type getNodeType() const { return m_type; }

    inline bool isInputOrBias() const { return m_type == Type::INPUT || m_type == Type::BIAS; }

protected:
    const Activation* m_activation = nullptr;
    float m_value = 0.f;
    Type m_type = Type::NONE;
};
