/*
* UnitTestBaseTypes.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <EvoAlgo/NeuralNetwork/Node.h>
#include <EvoAlgo/NeuralNetwork/Edge.h>

// Basic node class.
struct Node
{
    Node() = default;
    Node(float value) : m_value(value) {}

    float getValue() const { return m_value; }
    void setValue(float value) { m_value = value; }

    float m_value = 0.f;
};

// Basic edge class.
struct Edge
{
    Edge() = default;
    Edge(NodeId inNode, NodeId outNode, float weight = 0.f) : m_inNode(inNode), m_outNode(outNode), m_weight(weight) {}

    NodeId getInNode() const { return m_inNode; }
    NodeId getOutNode() const { return m_outNode; }
    float getWeight() const { return m_weight; }
    void setWeight(float weight) { m_weight = weight; }
    bool isEnabled() const { return true; }
    void copyState(const Edge* other) { m_weight = other->m_weight; }

    NodeId m_inNode = NodeId::invalid();
    NodeId m_outNode = NodeId::invalid();
    float m_weight = 0.f;
};
