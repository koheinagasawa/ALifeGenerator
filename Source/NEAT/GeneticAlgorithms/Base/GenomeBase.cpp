/*
* GenomeBase.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/Base/GenomeBase.h>


GenomeBase::GenomeBase(const GenomeBase& other)
    : m_biasNode(other.m_biasNode)
{
    // Copy the network
    m_network = other.m_network->clone();
}

void GenomeBase::operator= (const GenomeBase& other)
{
    m_biasNode = other.m_biasNode;

    // Copy the network
    m_network = other.m_network->clone();
}

std::shared_ptr<GenomeBase> GenomeBase::clone() const
{
    return std::make_shared<GenomeBase>(*this);
}

int GenomeBase::getNumEnabledEdges() const
{
    int num = 0;
    for (auto& itr : m_network->getEdges())
    {
        if (itr.second.isEnabled())
        {
            num++;
        }
    }
    return num;
}

void GenomeBase::clearNodeValues() const
{
    for (auto itr : m_network->getNodes())
    {
        m_network->accessNode(itr.first).setValue(0.f);
    }
}

void GenomeBase::setInputNodeValues(const std::vector<float>& values, float biasNodeValue) const
{
    assert(values.size() == m_network->getInputNodes().size());

    // Set bias node value.
    if (m_biasNode.isValid())
    {
        setBiasNodeValue(biasNodeValue);
    }

    // Set input node values.
    for (int i = 0; i < (int)values.size(); i++)
    {
        m_network->setNodeValue(m_network->getInputNodes()[i], values[i]);
    }
}

void GenomeBase::setBiasNodeValue(float value) const
{
    if (!m_biasNode.isValid())
    {
        WARN("No bias node in this genome");
        return;
    }

    m_network->accessNode(m_biasNode).setValue(value);
}

void GenomeBase::setActivation(NodeId nodeId, const Activation* activation)
{
    assert(m_network.get());
    assert(!m_network->getNode(nodeId).isInputOrBias());

    m_network->accessNode(nodeId).setActivation(activation);
}

void GenomeBase::setActivationAll(const Activation* activation)
{
    assert(m_network.get());

    // Set activation for all hidden and output nodes.
    for (auto itr : m_network->getNodes())
    {
        Node& node = m_network->accessNode(itr.first);
        const Node::Type nodeType = node.getNodeType();
        if (nodeType == Node::Type::HIDDEN || nodeType == Node::Type::OUTPUT)
        {
            node.setActivation(activation);
        }
    }
}

void GenomeBase::evaluate() const
{
    assert(m_network.get());
    m_network->evaluate();
}
