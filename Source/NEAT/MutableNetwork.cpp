/*
* NeuralNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/MutableNetwork.h>

InnovationId InnovationCounter::s_id(0);
InnovationCounter InnovationCounter::s_instance;

InnovationId InnovationCounter::getNewInnovationId()
{
    InnovationId idOut = s_id;
    s_id = s_id.val() + 1;
    return idOut;
}

SwitchableEdge::SwitchableEdge(NodeId inNode, NodeId outNode, float weight)
    : m_inNode(inNode)
    , m_outNode(outNode)
    , m_weight(weight)
    , m_enabled(true)
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
