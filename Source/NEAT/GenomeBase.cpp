/*
* GenomeBase.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GenomeBase.h>

GenomeBase::Node::Node(Type type)
    : m_type(type)
{
}

float GenomeBase::Node::getValue() const
{
    return m_value;
}

void GenomeBase::Node::setValue(float value)
{
    m_value = m_activation ? m_activation->activate(value) : value;
}

GenomeBase::GenomeBase(const Activation* defaultActivation)
    : m_defaultActivation(defaultActivation)
{
}

GenomeBase::GenomeBase(const GenomeBase& other)
    : m_inputNodes(other.m_inputNodes)
    , m_defaultActivation(other.m_defaultActivation)
{
    // Copy the network
    m_network = std::make_shared<Network>(*other.m_network.get());
}

void GenomeBase::operator= (const GenomeBase& other)
{
    m_inputNodes = other.m_inputNodes;
    m_defaultActivation = other.m_defaultActivation;

    // Copy the network
    m_network = std::make_shared<Network>(*other.m_network.get());
}

void GenomeBase::setInputNodeValues(const std::vector<float>& values) const
{
    assert(values.size() == m_inputNodes.size());

    for (int i = 0; i < (int)values.size(); i++)
    {
        m_network->setNodeValue(m_inputNodes[i], values[i]);
    }
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

void GenomeBase::evaluate(const std::vector<float>& inputNodeValues) const
{
    setInputNodeValues(inputNodeValues);
    evaluate();
}

void GenomeBase::evaluate() const
{
    assert(m_network.get());
    m_network->evaluate();
}

void GenomeBase::setNodeTypeAndActivation(NodeId nodeId, Node::Type type, const Activation* activation)
{
    Node& node = m_network->accessNode(nodeId);
    node.m_type = type;
    node.m_activation = activation;
}
