/*
* Edge.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/BaseType.h>
#include <EvoAlgo/NeuralNetwork/Node.h>

DECLARE_ID(EdgeId);

// Default edge structure.
struct DefaultEdge
{
    // Default constructor. This is used only by container of Edge in Network class and users shouldn't call it.
    DefaultEdge() = default;

    // Constructor using inNode and outNode ids.
    DefaultEdge(NodeId inNode, NodeId outNode, float weight = 1.f, bool enabled = true);

    // Copy and move constructor and operator
    DefaultEdge(const DefaultEdge& other) = default;
    DefaultEdge(DefaultEdge&& other) = default;
    void operator=(const DefaultEdge& other);
    void operator=(DefaultEdge&& other);

    inline bool isEnabled() const { return m_enabled; }
    inline void setEnabled(bool enable) { m_enabled = enable; }
    inline float getWeightRaw() const { return m_weight; }

    inline NodeId getInNode() const { return m_inNode; }
    inline NodeId getOutNode() const { return m_outNode; }
    inline void setWeight(float weight) { m_weight = weight; }
    inline float getWeight() const { return m_enabled ? m_weight : 0.f; }

    // Copy internal state of the edge (e.g. weight) without copying in node and out node ids.
    void copyState(const DefaultEdge* other);

protected:
    const NodeId m_inNode, m_outNode;
    float m_weight = 0.f;
    bool m_enabled = false;
};
