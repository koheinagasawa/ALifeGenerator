/*
* NeuralNetwork.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

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
