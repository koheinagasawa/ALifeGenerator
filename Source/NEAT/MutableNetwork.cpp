/*
* NeuralNetwork.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/MutableNetwork.h>

SwitchableEdge::SwitchableEdge(NodeId inNode, NodeId outNode, float weight, bool enabled)
    : m_inNode(inNode)
    , m_outNode(outNode)
    , m_weight(weight)
    , m_enabled(enabled)
{
}

SwitchableEdge::SwitchableEdge()
    : m_enabled(false)
{
}

void SwitchableEdge::operator=(const SwitchableEdge& other)
{
    *(const_cast<NodeId*>(&m_inNode)) = other.m_inNode;
    *(const_cast<NodeId*>(&m_outNode)) = other.m_outNode;
    m_weight = other.m_weight;
    m_enabled = other.m_enabled;
}

void SwitchableEdge::operator=(SwitchableEdge&& other)
{
    *(const_cast<NodeId*>(&m_inNode)) = std::move(other.m_inNode);
    *(const_cast<NodeId*>(&m_outNode)) = std::move(other.m_outNode);
    m_weight = other.m_weight;
    m_enabled = other.m_enabled;
}

NodeId SwitchableEdge::getInNode() const
{
    return m_inNode;
}

NodeId SwitchableEdge::getOutNode() const
{
    return m_outNode;
}

float SwitchableEdge::getWeight() const
{
    return m_enabled ? m_weight : 0.f;
}

void SwitchableEdge::setWeight(float weight)
{
    m_weight = weight;
}
