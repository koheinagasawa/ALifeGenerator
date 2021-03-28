/*
* GenomeTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/GeneticAlgorithms/NEAT/Genome.h>
#include <NEAT/GeneticAlgorithms/NEAT/Modifiers/DefaultMutation.h>

TEST(Genome, CreateGenome)
{
    using namespace NEAT;

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome genome(cinfo);

    const Genome::Network* network = genome.getNetwork();

    // Verify the genome's structure. It should be fully connected network with two input nodes and two output nodes.
    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNode(NodeId(0)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network->getNode(NodeId(1)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network->getNode(NodeId(2)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network->getNode(NodeId(3)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_EQ(genome.getInnovations().size(), 4);

    // Create another genome by copying the original one.
    Genome genome2(genome);

    const Genome::Network* network2 = genome.getNetwork();

    // Check that genome2 is identical with genome1.
    EXPECT_TRUE(genome2.validate());
    EXPECT_EQ(genome2.getInputNodes().size(), 2);
    EXPECT_EQ(network2->getNumNodes(), 4);
    EXPECT_EQ(network2->getNode(NodeId(0)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network2->getNode(NodeId(1)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network2->getNode(NodeId(2)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network2->getNode(NodeId(3)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network2->getNumEdges(), 4);
    EXPECT_EQ(network2->getOutputNodes().size(), 2);
    EXPECT_EQ(genome2.getInnovations().size(), 4);

    // Check if innovation ids are the same
    for (int i = 0; i < (int)genome.getInnovations().size(); i++)
    {
        const EdgeId e1 = genome.getInnovations()[i];
        const EdgeId e2 = genome2.getInnovations()[i];
        EXPECT_EQ(e1, e2);
    }
}

TEST(Genome, ModifyGenome)
{
    using namespace NEAT;

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome genome(cinfo);

    const Genome::Network* network = genome.getNetwork();

    // Verify the genome's structure. It should be fully connected network with two input nodes and two output nodes.
    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNode(NodeId(0)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network->getNode(NodeId(1)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network->getNode(NodeId(2)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network->getNode(NodeId(3)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_EQ(genome.getInnovations().size(), 4);

    NodeId newNode;
    EdgeId newEdge1, newEdge2, newEdge3;

    // Add a new node
    genome.addNodeAt(EdgeId(0), newNode, newEdge1, newEdge2);
    EXPECT_NE(newNode, NodeId::invalid());
    EXPECT_NE(newEdge1, EdgeId::invalid());
    EXPECT_NE(newEdge2, EdgeId::invalid());
    EXPECT_EQ(network->getNumNodes(), 5);
    EXPECT_EQ(network->getNumEdges(), 6);
    EXPECT_EQ(genome.getInnovations().size(), 6);
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_TRUE(network->hasNode(newNode));
    EXPECT_TRUE(network->hasEdge(newEdge1));
    EXPECT_TRUE(network->hasEdge(newEdge2));

    // Try to add an edge at already connected nodes
    EXPECT_TRUE(network->isConnected(NodeId(1), NodeId(3)));
    newEdge3 = genome.addEdgeAt(NodeId(1), NodeId(3), 3.f);
    EXPECT_EQ(newEdge3, EdgeId::invalid());

    // Try to add an edge
    EXPECT_FALSE(network->isConnected(NodeId(3), newNode));
    newEdge3 = genome.addEdgeAt(NodeId(3), newNode, 3.f);
    EXPECT_NE(newEdge3, EdgeId::invalid());
    EXPECT_TRUE(network->isConnected(NodeId(3), newNode));
    EXPECT_EQ(network->getNumNodes(), 5);
    EXPECT_EQ(network->getNumEdges(), 7);
    EXPECT_EQ(genome.getInnovations().size(), 7);
    EXPECT_TRUE(network->hasEdge(newEdge3));
    EXPECT_EQ(network->getWeight(newEdge3), 3.f);
    EXPECT_TRUE(network->isEdgeEnabled(newEdge3));
}

TEST(Genome, ReassignInnovation)
{
    using namespace NEAT;

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome genome(cinfo);

    const Genome::Network* network = genome.getNetwork();

    // Verify the genome's structure. It should be fully connected network with two input nodes and two output nodes.
    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNode(NodeId(0)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network->getNode(NodeId(1)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network->getNode(NodeId(2)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network->getNode(NodeId(3)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_EQ(genome.getInnovations().size(), 4);

    // Reassign innovation id of an edge.
    EdgeId originalEdge(0);
    EdgeId newEdge(4);
    NodeId outNode1(2);
    EXPECT_TRUE(network->hasEdge(originalEdge));
    EXPECT_FALSE(network->hasEdge(newEdge));
    EXPECT_EQ(network->getIncomingEdges(outNode1)[0], originalEdge);
    genome.reassignInnovation(originalEdge, newEdge);
    EXPECT_FALSE(network->hasEdge(originalEdge));
    EXPECT_TRUE(network->hasEdge(newEdge));
    EXPECT_EQ(network->getIncomingEdges(outNode1)[0], newEdge);
}

TEST(Genome, ReassignNodeId)
{
    using namespace NEAT;

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome genome(cinfo);

    const Genome::Network* network = genome.getNetwork();

    // Verify the genome's structure. It should be fully connected network with two input nodes and two output nodes.
    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNode(NodeId(0)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network->getNode(NodeId(1)).getNodeType(), Genome::Node::Type::INPUT);
    EXPECT_EQ(network->getNode(NodeId(2)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network->getNode(NodeId(3)).getNodeType(), Genome::Node::Type::OUTPUT);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_EQ(genome.getInnovations().size(), 4);

    // Reassign NodeId of a node.
    NodeId originalNode(0);
    NodeId newNode(4);
    EdgeId edge(0);
    EXPECT_TRUE(network->hasNode(originalNode));
    EXPECT_FALSE(network->hasNode(newNode));
    EXPECT_EQ(network->getInNode(edge), originalNode);
    EXPECT_EQ(genome.getInputNodes()[0], originalNode);
    genome.reassignNodeId(originalNode, newNode);
    EXPECT_FALSE(network->hasNode(originalNode));
    EXPECT_TRUE(network->hasNode(newNode));
    EXPECT_EQ(network->getInNode(edge), newNode);
    EXPECT_EQ(genome.getInputNodes()[0], newNode);
}

TEST(Genome, EvaluateGenome)
{
    using namespace NEAT;

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome::Activation activation = [](float value) { return value * 2.f; };
    cinfo.m_defaultActivation = &activation;
    Genome genome(cinfo);

    const Genome::Network::NodeIds& outputNodes = genome.getNetwork()->getOutputNodes();

    // Evaluate the network
    std::vector<float> inputs;
    inputs.push_back(1.f);
    inputs.push_back(2.f);
    genome.evaluate(inputs);

    // Check the node values are expected.
    for (NodeId nodeId : outputNodes)
    {
        EXPECT_EQ(genome.getNetwork()->getNode(nodeId).getValue(), 6.f);
    }

    // Change an edge weight
    genome.setEdgeWeight(EdgeId(0), 0.5f);

    // Change activation
    Genome::Activation activation2([](float value) { return value >= 3.f ? 1.f : 0.f; });
    genome.setActivationAll(&activation2);

    // Evaluate the network again.
    genome.evaluate();

    // Check the node values are expected.
    EXPECT_EQ(genome.getNetwork()->getNode(outputNodes[0]).getValue(), 0.f);
    EXPECT_EQ(genome.getNetwork()->getNode(outputNodes[1]).getValue(), 1.f);
}

TEST(Genome, CalcGenomesDistance)
{
    using namespace NEAT;

    // Custom random generator which always selects the minimum integer.
    class MyRandom : public PseudoRandom
    {
    public:
        MyRandom() : PseudoRandom(0) {}
        virtual int randomInteger(int min, int max) override { return min; }
    };

    // Create two genomes.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome genome1(cinfo);
    Genome genome2(genome1);

    // Set edge weights.
    {
        int count = 0;
        for (auto& itr : genome1.getNetwork()->getEdges())
        {
            float weight1 = (float)count;
            genome1.setEdgeWeight(itr.first, weight1);
            float weight2 = (float)(count + 4);
            genome2.setEdgeWeight(itr.first, weight2);
            count++;
        }
    }

    // Mutate the genomes several times.
    {
        MyRandom random;
        DefaultMutation mutator;
        mutator.m_params.m_weightMutationRate = 0.0f;
        mutator.m_params.m_addEdgeMutationRate = 0.0f;
        mutator.m_params.m_addNodeMutationRate = 1.0f;
        mutator.m_params.m_random = &random;

        DefaultMutation::MutationOut mutOut;

        mutator.mutate(&genome1, mutOut);
        EXPECT_EQ(mutOut.m_numNodesAdded, 1);
        EXPECT_EQ(mutOut.m_numEdgesAdded, 2);
        mutator.m_params.m_addEdgeMutationRate = 1.0f;
        mutator.mutate(&genome1, mutOut);
        EXPECT_EQ(mutOut.m_numNodesAdded, 1);
        EXPECT_EQ(mutOut.m_numEdgesAdded, 3);

        EXPECT_TRUE(genome1.validate());
        EXPECT_EQ(genome1.getNetwork()->getNumNodes(), 6);
        EXPECT_EQ(genome1.getNetwork()->getNumEdges(), 9);

        mutator.m_params.m_addEdgeMutationRate = 0.0f;
        mutator.mutate(&genome2, mutOut);
        EXPECT_EQ(mutOut.m_numNodesAdded, 1);
        EXPECT_EQ(mutOut.m_numEdgesAdded, 2);

        EXPECT_TRUE(genome2.validate());
        EXPECT_EQ(genome2.getNetwork()->getNumNodes(), 5);
        EXPECT_EQ(genome2.getNetwork()->getNumEdges(), 6);
    }

    // Calculate the distance of the two genomes.
    Genome::CalcDistParams params;
    params.m_disjointFactor = 0.5f;
    params.m_weightFactor = 0.25f;

    EXPECT_EQ(Genome::calcDistance(genome1, genome1, params), 0.f);
    EXPECT_EQ(Genome::calcDistance(genome1, genome2, params), 4.3125f); // 7 * 0.5 + (0 + 5 + 4 + 4) / 4 * 0.25 <- note that some edges were disabled by mutation.
}
