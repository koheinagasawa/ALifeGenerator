/*
* Edge.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/BaseType.h>
#include <NEAT/NeuralNetwork/Node.h>

DECLARE_ID(EdgeId);

// Base struct of edge.
struct EdgeBase
{
    virtual NodeId getInNode() const = 0;
    virtual NodeId getOutNode() const = 0;
    virtual float getWeight() const = 0;
    virtual void setWeight(float weight) = 0;

    virtual bool isEnabled() const { return true; }

    // Copy internal state of the edge (e.g. weight) without copying in node and out node ids.
    virtual void copyState(const EdgeBase* other) {}
};

// Default edge structure.
struct DefaultEdge : public EdgeBase
{
    // Default constructor. This is used only by container of Edge in Network class and users shouldn't call it.
    DefaultEdge() = default;

    // Constructor using inNode and outNode ids.
    DefaultEdge(NodeId inNode, NodeId outNode, float weight = 1.f);

    // Copy and move constructor and operator
    DefaultEdge(const DefaultEdge& other) = default;
    DefaultEdge(DefaultEdge&& other) = default;
    void operator=(const DefaultEdge& other);
    void operator=(DefaultEdge&& other);

    // Copy internal state of the edge (e.g. weight) without copying in node and out node ids.
    virtual void copyState(const EdgeBase* other) override;

    virtual NodeId getInNode() const override;
    virtual NodeId getOutNode() const override;
    virtual float getWeight() const override;
    virtual void setWeight(float weight) override;

protected:
    const NodeId m_inNode, m_outNode;
    float m_weight = 0.f;
};

// Edge which can be turned on and off without losing previous weight value.
struct SwitchableEdge : public DefaultEdge
{
    // Default constructor. This is used only by container of Edge in Network class and users shouldn't call it.
    SwitchableEdge();

    // Constructor using inNode and outNode ids.
    SwitchableEdge(NodeId inNode, NodeId outNode, float weight = 1.f, bool enabled = true);

    // Copy and move constructor and operator
    SwitchableEdge(const SwitchableEdge& other) = default;
    SwitchableEdge(SwitchableEdge&& other) = default;
    void operator=(const SwitchableEdge& other);
    void operator=(SwitchableEdge&& other);

    // Copy internal state of the edge (e.g. weight) without copying in node and out node ids.
    virtual void copyState(const EdgeBase* other) override;

    virtual bool isEnabled() const override { return m_enabled; }
    inline void setEnabled(bool enable) { m_enabled = enable; }

    // Return the weight. Return 0 if this edge is disabled.
    virtual float getWeight() const override;

    // Return the weight regardless of whether this edge is enabled.
    inline float getWeightRaw() const { return m_weight; }

protected:
    bool m_enabled = false;
};
