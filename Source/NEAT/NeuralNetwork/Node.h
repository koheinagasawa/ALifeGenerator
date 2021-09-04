/*
* Node.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/Activations/Activation.h>

DECLARE_ID(NodeId);

// Default node structure.
struct DefaultNode
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

    inline float getValue() const { return m_activation ? m_activation->activate(m_value) : m_value; }
    inline float getRawValue() const { return m_value; }
    inline void setValue(float value) { m_value = value; }

    inline auto getActivation() const->const Activation* { return m_activation; }
    inline void setActivation(const Activation* activation) { m_activation = activation; }
    inline auto getActivationName() const->const char* { return m_activation ? m_activation->m_name : nullptr; }
    inline auto getActivationId() const->ActivationId { return m_activation ? m_activation->m_id : ActivationId::invalid(); }

    inline void setNodeType(Type type) { m_type = type; }
    inline Type getNodeType() const { return m_type; }

    inline bool isInputOrBias() const { return m_type == Type::INPUT || m_type == Type::BIAS; }

protected:
    const Activation* m_activation = nullptr;
    float m_value = 0.f;
    Type m_type = Type::NONE;
};
