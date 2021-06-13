/*
* Edge.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/NeuralNetwork/Edge.h>

DefaultEdge::DefaultEdge(NodeId inNode, NodeId outNode, float weight)
    : m_inNode(inNode)
    , m_outNode(outNode)
    , m_weight(weight)
{
}

void DefaultEdge::operator=(const DefaultEdge& other)
{
    *(const_cast<NodeId*>(&m_inNode)) = other.m_inNode;
    *(const_cast<NodeId*>(&m_outNode)) = other.m_outNode;
    m_weight = other.m_weight;
}

void DefaultEdge::operator=(DefaultEdge&& other)
{
    *(const_cast<NodeId*>(&m_inNode)) = other.m_inNode;
    *(const_cast<NodeId*>(&m_outNode)) = other.m_outNode;
    m_weight = other.m_weight;
}

void DefaultEdge::copyState(const EdgeBase* other)
{
    m_weight = other->getWeight();
}

NodeId DefaultEdge::getInNode() const
{
    return m_inNode;
}

NodeId DefaultEdge::getOutNode() const
{
    return m_outNode;
}

float DefaultEdge::getWeight() const
{
    return m_weight;
}

void DefaultEdge::setWeight(float weight)
{
    m_weight = weight;
}

SwitchableEdge::SwitchableEdge(NodeId inNode, NodeId outNode, float weight, bool enabled)
    : DefaultEdge(inNode, outNode, weight)
    , m_enabled(enabled)
{
}

SwitchableEdge::SwitchableEdge()
    : m_enabled(false)
{
}

void SwitchableEdge::operator=(const SwitchableEdge& other)
{
    this->DefaultEdge::operator=(other);
    m_enabled = other.m_enabled;
}

void SwitchableEdge::operator=(SwitchableEdge&& other)
{
    this->DefaultEdge::operator=(other);
    m_enabled = other.m_enabled;
}

void SwitchableEdge::copyState(const EdgeBase* other)
{
    DefaultEdge::copyState(other);
    m_enabled = static_cast<const SwitchableEdge*>(other)->m_enabled;
}

float SwitchableEdge::getWeight() const
{
    return m_enabled ? m_weight : 0.f;
}
