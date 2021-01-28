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
    
    // All the weight should be 1.0
    const Genome::Network::Edges& edges = network->getEdges();
    for (auto& itr : edges)
    {
        EXPECT_EQ(network->getWeight(itr.first), 1.0);
    }

    // Let add node/edge mutation happen all the time
    Genome::MutationParams params;
    params.m_weightMutationRate = 0.0f;
    params.m_addEdgeMutationRate = 1.0f;
    params.m_addNodeMutationRate = 1.0f;

    Genome::MutationOut out;

    // Mutate the genome.
    // Edges are full connected already so we shouldn't be able to add new edge.
    // A new node should be added and as a result the number of edge should be increased too.
    genome.mutate(params, out);

    EXPECT_TRUE(network->validate());
    EXPECT_EQ(network->getNumNodes(), 5);
    EXPECT_EQ(network->getNode(NodeId(4)).getNodeType(), Genome::Node::Type::HIDDEN);
    EXPECT_EQ(network->getNumEdges(), 5);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_TRUE(out.m_newEdges[0].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdges[0].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdges[0].m_newEdge.isValid());
    EXPECT_FALSE(out.m_newEdges[1].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdges[1].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdges[1].m_newEdge.isValid());

    // Mutate the genome again.
    // Now we should be able to add both new node and edge.
    // So the number of nodes is +1 and the number of edges is +2
    genome.mutate(params, out);

    EXPECT_TRUE(network->validate());
    EXPECT_EQ(network->getNumNodes(), 6);
    EXPECT_EQ(network->getNode(NodeId(5)).getNodeType(), Genome::Node::Type::HIDDEN);
    EXPECT_EQ(network->getNumEdges(), 7);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_TRUE(out.m_newEdges[0].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdges[0].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdges[0].m_newEdge.isValid());
    EXPECT_TRUE(out.m_newEdges[1].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdges[1].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdges[1].m_newEdge.isValid());

    // Reset parameter so that no mutation should happen
    params.m_addEdgeMutationRate = 0.0f;
    params.m_addNodeMutationRate = 0.0f;

    genome.mutate(params, out);

    EXPECT_TRUE(network->validate());
    EXPECT_EQ(network->getNumNodes(), 6);
    EXPECT_EQ(network->getNumEdges(), 7);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_FALSE(out.m_newEdges[0].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdges[0].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdges[0].m_newEdge.isValid());
    EXPECT_FALSE(out.m_newEdges[1].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdges[1].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdges[1].m_newEdge.isValid());

    // Mutate only edge weights
    {
        params.m_weightMutationRate = 1.0f;
        params.m_weightMutationNewValRate = 0.0f;
        const float perturbation = 0.1f;
        params.m_weightMutationPerturbation = perturbation;

        // Remember original edge weights
        std::unordered_map<EdgeId, float> originalWeights;
        for (auto& itr : edges)
        {
            originalWeights[itr.first] = network->getWeight(itr.first);
        }

        genome.mutate(params, out);

        for (auto& itr : edges)
        {
            float original = originalWeights.at(itr.first);
            float weight = network->getWeight(itr.first);
            EXPECT_TRUE((original * weight) > 0); // Check weight hasn't changed its sign.
            original = std::abs(original);
            weight = std::abs(weight);
            EXPECT_TRUE(weight >= (original * (1.f - perturbation)) && weight <= (original * (1.f + perturbation)));
        }
    }

    // Mutate edge weights by a new value all the time.
    {
        // Custom random generator which returns 3.f all the time.
        class CustomRandom : public PseudoRandom
        {
        public:
            CustomRandom() : PseudoRandom(0) {}
            float randomReal(float min, float max) override { return 3.0f; }
            float randomReal01() override { return 0; }
        };

        CustomRandom random;
        params.m_weightMutationNewValRate = 1.0f;
        params.m_random = &random;

        genome.mutate(params, out);

        for (auto& itr : edges)
        {
            EXPECT_EQ(network->getWeight(itr.first), 3.f);
        }
    }
}
