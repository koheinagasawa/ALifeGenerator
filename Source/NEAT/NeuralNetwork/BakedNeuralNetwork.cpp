/*
* BakedNeuralNetwork.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/NeuralNetwork/BakedNeuralNetwork.h>
#include <NEAT/NeuralNetwork/NeuralNetwork.h>

const std::function<float(float)> BakedNeuralNetwork::s_nullActivation = [](float val) { return val; };

BakedNeuralNetwork::BakedNeuralNetwork(const Network* network)
    : m_isCircularNetwork(network->hasCircularEdges())
{
    m_nodes.reserve(network->getNodes().size());
    m_edges.reserve(network->getEdges().size());

    const Network::NodeIds& outputNodes = network->getOutputNodes();

    std::unordered_set<NodeId> addedNodes;
    std::vector<NodeId> stack;

    // Iterate over output nodes and collect any nodes and edges connected to those output nodes.
    for (NodeId outputNodeId : outputNodes)
    {
        stack.clear();

        // Recursively visit connected nodes
        stack.push_back(outputNodeId);
        while (stack.size() > 0)
        {
            NodeId id = stack.back();

            if (addedNodes.find(id) != addedNodes.end())
            {
                // We visited this node already. Skip.
                stack.pop_back();
                continue;
            }

            bool canAddToList = true;

            // Iterate over connected nodes while counting enabled edges.
            const Network::EdgeIds& incomingEdges = network->getIncomingEdges(id);
            int numEdges = 0;

            for (EdgeId incomingId : incomingEdges)
            {
                const DefaultEdge& edge = network->getEdge(incomingId);

                // When the edge is disabled, returned weight should be zero.
                if (edge.getWeight() == 0.f)
                {
                    continue;
                }

                numEdges++;
                NodeId inNodeId = edge.getInNode();

                bool isNewNode = true;
                if (m_isCircularNetwork)
                {
                    for (NodeId i : stack)
                    {
                        if (i == inNodeId)
                        {
                            isNewNode = false;
                            break;
                        }
                    }
                }

                // Recurse if we haven't evaluated this in node yet.
                if (isNewNode && addedNodes.find(inNodeId) == addedNodes.end())
                {
                    stack.push_back(inNodeId);
                    canAddToList = false;
                    continue;
                }
            }

            if (canAddToList)
            {
                // All nodes connected from inputs to this node should be added to the list already.
                // Now we can add this node to the list.

                assert(addedNodes.find(id) == addedNodes.end());

                // Create node data
                Node entry;
                entry.m_startEdge = (int)m_edges.size();
                entry.m_numEdges = (unsigned short)numEdges;

                // Create edge data
                for (EdgeId incomingId : incomingEdges)
                {
                    const DefaultEdge& edge = network->getEdge(incomingId);

                    if (edge.getWeight() == 0.f)
                    {
                        continue;
                    }

                    assert(m_nodeIdIndexMap.find(edge.getInNode()) != m_nodeIdIndexMap.end());
                    m_edges.push_back(Edge{ m_nodeIdIndexMap.at(edge.getInNode()), edge.getWeight() });
                }

                const DefaultNode& node = network->getNode(id);

                // Set activation func index.
                {
                    // Get activation function.
                    ActivationFunc func;
                    const Activation* activation = node.getActivation();
                    if (activation == nullptr)
                    {
                        func = &s_nullActivation;
                    }
                    else
                    {
                        func = &activation->m_func;
                    }

                    // Check if this activation has already appeared and if so set its index.
                    int i = 0;
                    for (i = 0; i < (int)m_activationFuncs.size(); i++)
                    {
                        if (m_activationFuncs[i] == func)
                        {
                            entry.m_activationFunc = i;
                            break;
                        }
                    }

                    // If this is a new activation function, create a new entry.
                    if (i == (int)m_activationFuncs.size())
                    {
                        entry.m_activationFunc = (unsigned short)m_activationFuncs.size();
                        m_activationFuncs.push_back(func);
                    }
                }

                // Set the initial value of the node
                entry.m_value = node.getRawValue();

                // Add the node to the list.
                m_nodeIdIndexMap[id] = (int)m_nodes.size();
                m_nodes.push_back(entry);
                addedNodes.insert(id);
                stack.pop_back();
            }
        }
    }
}

void BakedNeuralNetwork::setNodeValue(NodeId node, float value)
{
    if (m_nodeIdIndexMap.find(node) == m_nodeIdIndexMap.end())
    {
        return;
    }

    int index = m_nodeIdIndexMap.at(node);
    assert(index >= 0 && index < (int)m_nodes.size());

    m_nodes[index].m_value = value;
    m_nodes[index].m_activatedValue = (*m_activationFuncs[m_nodes[index].m_activationFunc])(value);
}

void BakedNeuralNetwork::clearNodeValues()
{
    for (Node& node : m_nodes)
    {
        node.m_value = 0.f;
        node.m_activatedValue = 0.f;
    }
}

float BakedNeuralNetwork::getNodeValue(NodeId node) const
{
    assert(m_nodeIdIndexMap.find(node) != m_nodeIdIndexMap.end());

    int index = m_nodeIdIndexMap.at(node);
    assert(index >= 0 && index < (int)m_nodes.size());

    return m_nodes[index].m_activatedValue;
}

void BakedNeuralNetwork::evaluate()
{
    // Just evaluate nodes from start to end since they are already sorted in that way.
    for (Node& node : m_nodes)
    {
        float valueSum = 0.f;
        if (node.m_numEdges == 0)
        {
            valueSum = node.m_value;
        }
        else
        {
            // Accumulate the value from incoming edges.
            const Edge* edges = &m_edges[node.m_startEdge];
            for (int j = 0; j < node.m_numEdges; j++)
            {
                const Edge& edge = edges[j];
                valueSum += m_nodes[edge.m_node].m_activatedValue * edge.m_weight;
            }
        }

        // Activate the value.
        ActivationFunc activation = m_activationFuncs[node.m_activationFunc];
        node.m_activatedValue = (*activation)(valueSum);
    }
}
