/*
* GenomeBaseTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/GeneticAlgorithms/Base/GenomeBase.h>

namespace
{
    GenomeBase::Activation s_activation = [](float value) { return value * 2.f; };

    // Custom implementation of Genome.
    class MyGenome : public GenomeBase
    {
    public:
        MyGenome() : GenomeBase(&s_activation)
        {
        }

        void createNetwork(const Network::Nodes& nodes, const Network::Edges& edges)
        {
            Network::NodeIds outputNodes;
            for (auto& itr : nodes)
            {
                if (itr.second.getNodeType() == Node::Type::OUTPUT)
                {
                    outputNodes.push_back(itr.first);
                }
                if (itr.second.getNodeType() == Node::Type::INPUT)
                {
                    m_inputNodes.push_back(itr.first);
                }
            }

            m_network = std::make_shared<Network>(nodes, edges, outputNodes);
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
    using Edge = MyGenome::Network::Edge;

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

        edges.insert({ EdgeId(0), Edge(NodeId(0), NodeId(2), 2.0f) });
        edges.insert({ EdgeId(1), Edge(NodeId(1), NodeId(2), 3.0f) });
        edges.insert({ EdgeId(2), Edge(NodeId(2), NodeId(3), 4.0f) });
    }

    // Create network.
    genome.createNetwork(nodes, edges);
    EXPECT_TRUE(genome.getNetwork() != nullptr);

    // Test edge interface.
    EXPECT_EQ(genome.getEdgeWeight(EdgeId(0)), 2.0f);
    genome.setEdgeWeight(EdgeId(1), 4.0f);
    EXPECT_EQ(genome.getEdgeWeight(EdgeId(1)), 4.0f);

    // Test node interface.
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(genome.getInputNodes()[0], NodeId(0));
    EXPECT_EQ(genome.getInputNodes()[1], NodeId(1));
    EXPECT_EQ(genome.getNetwork()->getNode(NodeId(0)).getValue(), 0.f);
    EXPECT_EQ(genome.getNetwork()->getNode(NodeId(1)).getValue(), 0.f);
    {
        std::vector<float> inputValues;
        inputValues.push_back(5.f);
        inputValues.push_back(6.f);
        genome.setInputNodeValues(inputValues);
        EXPECT_EQ(genome.getNetwork()->getNode(NodeId(0)).getValue(), 5.f);
        EXPECT_EQ(genome.getNetwork()->getNode(NodeId(1)).getValue(), 6.f);
    }

    // Test activation interface
    EXPECT_EQ(genome.getDefaultActivation(), &s_activation);
    GenomeBase::Activation newActivation = [](float value) { return value; };
    genome.setDefaultActivation(&newActivation);
    EXPECT_EQ(genome.getDefaultActivation(), &newActivation);
    genome.setActivationAll(&s_activation);
    genome.setActivation(NodeId(3), &newActivation);

    // Test evaluation
    genome.evaluate();
    EXPECT_EQ(genome.getNetwork()->getNode(NodeId(3)).getValue(), 272.f);
    std::vector<float> inputValues;
    inputValues.push_back(1.f);
    inputValues.push_back(2.f);
    genome.evaluate(inputValues);
    EXPECT_EQ(genome.getNetwork()->getNode(NodeId(3)).getValue(), 80.f);
}
