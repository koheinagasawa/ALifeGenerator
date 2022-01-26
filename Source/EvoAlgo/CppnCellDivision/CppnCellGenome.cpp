/*
* CppnCellGenome.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <EvoAlgo/EvoAlgo.h>
#include <EvoAlgo/CppnCellDivision/CppnCellGenome.h>
#include <EvoAlgo/NeuralNetwork/NeuralNetworkFactory.h>

CppnCreatureGenome::CppnCreatureGenome(const Cinfo& cinfo)
{
    assert(cinfo.m_numInitialHiddenLayers == (int)cinfo.m_numNodeInInitialHiddenLayers.size());

    constexpr int numInputNodes = (int)InputNode::NUM_INPUT_NODES;
    constexpr int numOutputNodes = (int)OutputNode::NUM_OUTPUT_NODES;

    // Calculate the number of nodes
    int numNodes = numInputNodes + numOutputNodes + 1; // +1 is for bias node
    for (int numNodeInHiddenLayer : cinfo.m_numNodeInInitialHiddenLayers)
    {
        numNodes += numNodeInHiddenLayer;
    }

    // Create buffers
    Network::Nodes nodes;
    Network::Edges edges;
    Network::NodeIds outputNodes;
    Network::NodeIds inputNodes;

    // Create nodes
    int nodeId = 0;
    nodes.reserve(numNodes);
    inputNodes.reserve(numInputNodes);
    outputNodes.reserve(numOutputNodes);

    // Create input nodes.
    for (int i = 0; i < numInputNodes; i++)
    {
        NodeId id = nodeId++;
        nodes.insert({ id, Node(Node::Type::INPUT) });
        inputNodes.push_back(id);
    }

    // Create hidden nodes
    for (int i = 0; i < cinfo.m_numInitialHiddenLayers; i++)
    {
        for (int j = 0; j < cinfo.m_numNodeInInitialHiddenLayers[i]; j++)
        {
            NodeId id = nodeId++;
            nodes.insert({ id, Node(Node::Type::HIDDEN) });
            if (cinfo.m_activationProvider)
            {
                nodes[id].setActivation(cinfo.m_activationProvider->getActivation());
            }
        }
    }

    // Create output nodes
    for (int i = 0; i < numOutputNodes; i++)
    {
        NodeId id = nodeId++;
        nodes.insert({ id, Node(Node::Type::OUTPUT) });
        if (cinfo.m_activationProvider)
        {
            nodes[id].setActivation(cinfo.m_activationProvider->getActivation());
        }
        outputNodes.push_back(id);
    }

    // Create a bias node
    {
        m_biasNode = nodeId++;
        nodes.insert({ m_biasNode, Node(Node::Type::BIAS) });
        nodes[m_biasNode].setValue(cinfo.m_biasNodeValue);
    }

    // Create fully connected network
    int edgeId = 0;
    int startL1Node = 0;
    for (int layer = 0; layer < cinfo.m_numInitialHiddenLayers + 1; layer++)
    {
        int numL1Nodes = (layer == 0) ? numInputNodes : cinfo.m_numNodeInInitialHiddenLayers[layer - 1];
        int numL2Nodes = (layer == cinfo.m_numInitialHiddenLayers) ? numOutputNodes : cinfo.m_numNodeInInitialHiddenLayers[layer];
        int startL2Node = startL1Node + numL1Nodes;

        // Fully connect each layer
        for (int i = 0, inNode = startL1Node; i < numL1Nodes; i++, inNode++)
        {
            for (int j = 0, outNode = startL2Node; j < numL2Nodes; j++, outNode++)
            {
                edges.insert({ EdgeId(edgeId++), Edge(inNode, outNode) });
            }
        }

        // Create edges from the bias node
        if (layer > 0)
        {
            for (int j = 0, outNode = startL2Node; j < numL2Nodes; j++, outNode++)
            {
                edges.insert({ EdgeId(edgeId++), Edge(m_biasNode, outNode) });
            }
        }

        startL1Node += numL1Nodes;
    }

    // Create the network
    m_network = NeuralNetworkFactory::createNeuralNetwork<Node, Edge>(NeuralNetworkType::FEED_FORWARD, nodes, edges, inputNodes, outputNodes);

    // Randomize edge weights
    if (cinfo.m_randomizeInitialEdges)
    {
        RandomGenerator* random;
        PseudoRandom defaultRandom(0);
        if(cinfo.m_randomWeightsGenerator)
        {
            random = cinfo.m_randomWeightsGenerator;
        }
        else
        {
            random = &defaultRandom;
        }

        for (const auto& elem : m_network->getEdges())
        {
            setEdgeWeight(elem.first, random->randomReal(cinfo.m_minWeight, cinfo.m_maxWeight));
        }
    }
}

bool CppnCreatureGenome::evaluateDivision(const std::vector<float>& inputNodeValues, Vector4& directionOut)
{
    // Set input values
    clearNodeValues();
    setInputNodeValues(inputNodeValues, 1.0f);

    // Evaluate the genome
    evaluate();

    const GenomeBase::Network::NodeIds& outputNodes = getOutputNodes();
    if (getNodeValue(outputNodes[(int)OutputNode::DEVIDE]) < 0.5f)
    {
        return false;
    }

    // Set direction of division
    directionOut.setComponent<0>(SimdFloat(getNodeValue(outputNodes[(int)OutputNode::DIRECTION_X])));
    directionOut.setComponent<1>(SimdFloat(getNodeValue(outputNodes[(int)OutputNode::DIRECTION_Y])));
    directionOut.setComponent<2>(SimdFloat(getNodeValue(outputNodes[(int)OutputNode::DIRECTION_Z])));
    directionOut.normalize<3>();

    // [TODO] Support orientation

    return true;
}