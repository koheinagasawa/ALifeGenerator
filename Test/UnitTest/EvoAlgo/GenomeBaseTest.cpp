/*
* GenomeBaseTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <EvoAlgo/GeneticAlgorithms/Base/GenomeBase.h>
#include <EvoAlgo/NeuralNetwork/FeedForwardNetwork.h>

namespace
{
    Activation s_activation = [](float value) { return value * 2.f; };

    // Custom implementation of Genome.
    class MyGenome : public GenomeBase
    {
    public:
        MyGenome() = default;

        void createNetwork(const Network::Nodes& nodes, const Network::Edges& edges)
        {
            Network::NodeIds inputNodes;
            Network::NodeIds outputNodes;
            for (auto& itr : nodes)
            {
                if (itr.second.getNodeType() == Node::Type::OUTPUT)
                {
                    outputNodes.push_back(itr.first);
                }
                else if (itr.second.getNodeType() == Node::Type::INPUT)
                {
                    inputNodes.push_back(itr.first);
                }
                else if (itr.second.getNodeType() == Node::Type::BIAS)
                {
                    m_biasNode = itr.first;
                }
            }

            m_network = std::make_shared<FeedForwardNetwork<Node, Edge>>(nodes, edges, inputNodes, outputNodes);
        }
    };
}

TEST(GenomeBase, GenomeBasicOperations)
{
    using Network = MyGenome::Network;
    using Nodes = Network::Nodes;
    using NodeIds = Network::NodeIds;
    using Edges = Network::Edges;
    using Node = MyGenome::Node;
    using Edge = MyGenome::Edge;

    // Create a genome.
    MyGenome genome;

    // Set up nodes and edges.
    Nodes nodes;
    Edges edges;
    {
        nodes.insert({ NodeId(0), Node(Node::Type::INPUT) });
        nodes.insert({ NodeId(1), Node(Node::Type::INPUT) });
        nodes.insert({ NodeId(2), Node(Node::Type::HIDDEN) });
        nodes.insert({ NodeId(3), Node(Node::Type::OUTPUT) });
        nodes.insert({ NodeId(4), Node(Node::Type::BIAS) });

        edges.insert({ EdgeId(0), Edge(NodeId(0), NodeId(2), 2.0f) });
        edges.insert({ EdgeId(1), Edge(NodeId(1), NodeId(2), 3.0f) });
        edges.insert({ EdgeId(2), Edge(NodeId(2), NodeId(3), 4.0f) });
    }

    // Create network.
    genome.createNetwork(nodes, edges);

    // Test edge interface.
    EXPECT_EQ(genome.getEdgeWeight(EdgeId(0)), 2.0f);
    genome.setEdgeWeight(EdgeId(1), 4.0f);
    EXPECT_EQ(genome.getEdgeWeight(EdgeId(1)), 4.0f);
    EXPECT_EQ(genome.getNumEnabledEdges(), 3);
    EXPECT_EQ(genome.getEdgeWeight(EdgeId(0)), 2.0f);
    EXPECT_EQ(genome.isEdgeEnabled(EdgeId(0)), true);
    genome.setEdgeEnabled(EdgeId(0), false);
    EXPECT_EQ(genome.isEdgeEnabled(EdgeId(0)), false);
    EXPECT_EQ(genome.getNumEnabledEdges(), 2);
    EXPECT_EQ(genome.getEdgeWeight(EdgeId(0)), 0.0f);
    EXPECT_EQ(genome.getEdgeWeightRaw(EdgeId(0)), 2.0f);
    genome.setEdgeEnabled(EdgeId(0), true);

    // Test node interface.
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(genome.getInputNodes()[0], NodeId(0));
    EXPECT_EQ(genome.getInputNodes()[1], NodeId(1));
    EXPECT_EQ(genome.getNodeValue(NodeId(0)), 0.f);
    EXPECT_EQ(genome.getNodeValue(NodeId(1)), 0.f);
    {
        std::vector<float> inputValues;
        inputValues.push_back(5.f);
        inputValues.push_back(6.f);
        genome.setInputNodeValues(inputValues);
        EXPECT_EQ(genome.getNodeValue(NodeId(0)), 5.f);
        EXPECT_EQ(genome.getNodeValue(NodeId(1)), 6.f);
    }
    EXPECT_EQ(genome.getBiasNode(), NodeId(4));
    EXPECT_EQ(genome.getNodeValue(genome.getBiasNode()), 0.f);
    genome.setBiasNodeValue(1.f);
    EXPECT_EQ(genome.getNodeValue(genome.getBiasNode()), 1.f);

    // Test activation interface
    Activation newActivation = [](float value) { return value; };
    genome.setActivationAll(&s_activation);
    genome.setActivation(NodeId(3), &newActivation);

    // Test evaluation
    genome.evaluate();
    EXPECT_EQ(genome.getNodeValue(NodeId(3)), 272.f); // (2 * (5 * 2 + 6 * 4)) * 4 = 272
    std::vector<float> inputValues;
    inputValues.push_back(1.f);
    inputValues.push_back(2.f);
    genome.clearNodeValues();
    genome.setInputNodeValues(inputValues);
    genome.evaluate();
    EXPECT_EQ(genome.getNodeValue(NodeId(3)), 80.f); // (2 * (1 * 2 + 2 * 4)) * 4 = 80

    // Clear node values
    genome.clearNodeValues();
    for (const auto& node : genome.getNetwork()->getNodes())
    {
        EXPECT_EQ(node.second.m_node.getValue(), 0);
    }
}
