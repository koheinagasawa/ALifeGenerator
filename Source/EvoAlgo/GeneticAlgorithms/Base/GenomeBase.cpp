/*
* GenomeBase.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <EvoAlgo/EvoAlgo.h>
#include <EvoAlgo/GeneticAlgorithms/Base/GenomeBase.h>
#include <EvoAlgo/NeuralNetwork/NeuralNetworkEvaluator.h>

GenomeBase::GenomeBase(NetworkPtr network, NodeId biasNode)
    : m_network(network)
    , m_biasNode(biasNode)
{
    assert(network->getInputNodes().size() > 0 && network->getOutputNodes().size() > 0);
    assert(!m_biasNode.isValid() || network->hasNode(biasNode));
}

GenomeBase::GenomeBase(const GenomeBase& other)
    : m_biasNode(other.m_biasNode)
    , m_needRebake(other.m_needRebake)
{
    // Copy the network
    m_network = other.m_network->clone();

    if (other.m_bakedNetwork)
    {
        m_bakedNetwork = std::make_shared<BakedNeuralNetwork>(*other.m_bakedNetwork);
    }
}

void GenomeBase::operator= (const GenomeBase& other)
{
    m_biasNode = other.m_biasNode;

    // Copy the network
    m_network = other.m_network->clone();

    if (other.m_bakedNetwork)
    {
        m_bakedNetwork = std::make_shared<BakedNeuralNetwork>(*other.m_bakedNetwork);
        m_needRebake = other.m_needRebake;
    }
    else
    {
        m_bakedNetwork = nullptr;
        m_needRebake = true;
    }
}

std::shared_ptr<GenomeBase> GenomeBase::clone() const
{
    return std::make_shared<GenomeBase>(*this);
}

void GenomeBase::bake()
{
    if (m_needRebake)
    {
        m_bakedNetwork = m_network->bake();
        m_needRebake = false;
    }
}

int GenomeBase::getNumEnabledEdges() const
{
    int num = 0;
    for (const auto& elem : m_network->getEdges())
    {
        if (elem.second.isEnabled())
        {
            num++;
        }
    }
    return num;
}

void GenomeBase::clearNodeValues()
{
    m_network->setAllNodeValues(0.f);

    if (shouldUpdateBakedNetworkNode())
    {
        m_bakedNetwork->clearNodeValues();
    }
}

void GenomeBase::setInputNodeValues(const std::vector<float>& values, float biasNodeValue)
{
    assert(values.size() == m_network->getInputNodes().size());

    // Set bias node value.
    if (m_biasNode.isValid())
    {
        setBiasNodeValue(biasNodeValue);
    }

    // Set input node values.
    const bool updateBakedNetwork = shouldUpdateBakedNetworkNode();
    for (int i = 0; i < (int)values.size(); i++)
    {
        NodeId nodeId = m_network->getInputNodes()[i];
        m_network->setNodeValue(nodeId, values[i]);
        if (updateBakedNetwork)
        {
            m_bakedNetwork->setNodeValue(nodeId, values[i]);
        }
    }
}

void GenomeBase::setBiasNodeValue(float value)
{
    if (!m_biasNode.isValid())
    {
        WARN("No bias node in this genome");
        return;
    }

    m_network->setNodeValue(m_biasNode, value);

    if (shouldUpdateBakedNetworkNode())
    {
        m_bakedNetwork->setNodeValue(m_biasNode, value);
    }
}

float GenomeBase::getNodeValue(NodeId nodeId) const
{
    if (shouldUpdateBakedNetworkNode())
    {
        return m_bakedNetwork->getNodeValue(nodeId);
    }

    return m_network->getNode(nodeId).getValue();
}

void GenomeBase::setActivation(NodeId nodeId, const Activation* activation)
{
    assert(m_network.get());
    assert(!m_network->getNode(nodeId).isInputOrBias());

    m_network->accessNode(nodeId).setActivation(activation);
    m_needRebake = true;
}

void GenomeBase::setActivationAll(const Activation* activation)
{
    assert(m_network.get());

    // Set activation for all hidden and output nodes.
    for (auto& elem : m_network->accessNodes())
    {
        Node& node = elem.second.m_node;
        const Node::Type nodeType = node.getNodeType();
        if (nodeType == Node::Type::HIDDEN || nodeType == Node::Type::OUTPUT)
        {
            node.setActivation(activation);
            m_needRebake = true;
        }
    }
}

void GenomeBase::evaluate()
{
    assert(m_network.get());

    bake();
    m_bakedNetwork->evaluate();
}

void GenomeBase::evaluate(NeuralNetworkEvaluator* evaluator)
{
    assert(m_network.get());

    if (!evaluator)
    {
        // Fallback to evaluation without evaluator.
        evaluate();
    }
    else
    {
        bake();
        evaluator->evaluate(m_network->getOutputNodes(), m_bakedNetwork.get());
    }
}
