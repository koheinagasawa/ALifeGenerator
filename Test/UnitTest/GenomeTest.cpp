/*
* GenomeTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/Genome.h>

TEST(Genome, CreateGenome)
{
    using namespace NEAT;

    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;

    // Create a genome.
    Genome genome(cinfo);

    const Genome::Network* network = genome.getNetwork();

    EXPECT_TRUE(network);
    EXPECT_TRUE(network->validate());
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_EQ(genome.getInnovations().size(), 4);

    // Create another genome by copying the original one.
    Genome genome2(genome);

    const Genome::Network* network2 = genome.getNetwork();

    EXPECT_TRUE(network2);
    EXPECT_TRUE(network2->validate());
    EXPECT_EQ(network2->getNumNodes(), 4);
    EXPECT_EQ(network2->getNumEdges(), 4);
    EXPECT_EQ(network2->getOutputNodes().size(), 2);
    EXPECT_EQ(genome2.getInnovations().size(), 4);

    // Check if innovation ids are the same
    for (int i = 0; i < (int)genome.getInnovations().size(); i++)
    {
        const Genome::InnovationEntry& entry1 = genome.getInnovations()[i];
        const Genome::InnovationEntry& entry2 = genome2.getInnovations()[i];
        EXPECT_EQ(entry1.m_id, entry2.m_id);
        EXPECT_EQ(entry1.m_edgeId, entry2.m_edgeId);
    }
}

TEST(Genome, MutateGenome)
{
    using namespace NEAT;

    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;

    // Create a genome.
    Genome genome(cinfo);

    const Genome::Network* network = genome.getNetwork();

    EXPECT_TRUE(network);
    EXPECT_TRUE(network->validate());
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

    // Let add node/edge mutation happen all the time
    Genome::MutationParams params;
    params.m_addEdgeMutationRate = 1.0f;
    params.m_addNodeMutationRate = 1.0f;

    Genome::MutationOut out;

    // Mutate the genome.
    // Edges are full connected already so we shouldn't be able to add new edge.
    // A new node should be added and as a result the number of edge should be increased too.
    genome.mutate(params, out);

    EXPECT_TRUE(network->validate());
    EXPECT_EQ(network->getNumNodes(), 5);
    EXPECT_EQ(network->getNumEdges(), 5);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

    // Mutate the genome again.
    // Now we should be able to add both new node and edge.
    // So the number of nodes is +1 and the number of edges is +2
    genome.mutate(params, out);

    EXPECT_TRUE(network->validate());
    EXPECT_EQ(network->getNumNodes(), 6);
    EXPECT_EQ(network->getNumEdges(), 7);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
}
