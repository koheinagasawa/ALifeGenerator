/*
* Edge.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <EvoAlgo/EvoAlgo.h>
#include <EvoAlgo/NeuralNetwork/Edge.h>

DefaultEdge::DefaultEdge(NodeId inNode, NodeId outNode, float weight, bool enabled)
    : m_inNode(inNode)
    , m_outNode(outNode)
    , m_weight(weight)
    , m_enabled(enabled)
{
}

void DefaultEdge::operator=(const DefaultEdge& other)
{
    *(const_cast<NodeId*>(&m_inNode)) = other.m_inNode;
    *(const_cast<NodeId*>(&m_outNode)) = other.m_outNode;
    m_weight = other.m_weight;
    m_enabled = other.m_enabled;
}

void DefaultEdge::operator=(DefaultEdge&& other)
{
    *(const_cast<NodeId*>(&m_inNode)) = other.m_inNode;
    *(const_cast<NodeId*>(&m_outNode)) = other.m_outNode;
    m_weight = other.m_weight;
    m_enabled = other.m_enabled;
}

void DefaultEdge::copyState(const DefaultEdge* other)
{
    m_weight = other->m_weight;
    m_enabled = other->m_enabled;
}

