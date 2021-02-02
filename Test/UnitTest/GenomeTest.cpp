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

    EXPECT_TRUE(genome.validate());
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

    EXPECT_TRUE(genome2.validate());
    EXPECT_EQ(network2->getNumNodes(), 4);
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

    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    
    // All the weight should be 1.0
    {
        const Genome::Network::Edges& edges = network->getEdges();
        for (auto& itr : edges)
        {
            EXPECT_EQ(network->getWeight(itr.first), 1.0);
        }
    }

    // Let add node/edge mutation happen all the time
    Genome::MutationParams params;
    params.m_weightMutationRate = 0.0f;
    params.m_addEdgeMutationRate = 1.0f;
    params.m_addNodeMutationRate = 1.0f;

    Genome::MutationOut out;

    // Mutate the genome.
    // Edges are full connected already so we shouldn't be able to add new edge.
    // A new node should be added and as a result the number of edge should be increased by 2 too.
    genome.mutate(params, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(network->getNumNodes(), 5);
    EXPECT_EQ(network->getNode(NodeId(4)).getNodeType(), Genome::Node::Type::HIDDEN);
    EXPECT_EQ(network->getNumEdges(), 6);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_EQ(out.m_numNodesAdded, 1);
    EXPECT_EQ(out.m_numEdgesAdded, 2);
    EXPECT_TRUE(out.m_newEdges[0].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdges[0].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdges[0].m_newEdge.isValid());
    EXPECT_TRUE(out.m_newEdges[1].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdges[1].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdges[1].m_newEdge.isValid());
    EXPECT_FALSE(out.m_newEdges[2].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdges[2].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdges[2].m_newEdge.isValid());

    // Mutate the genome again.
    // Now we should be able to add both new node and edge.
    // So the number of nodes is +1 and the number of edges is +3
    genome.mutate(params, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(network->getNumNodes(), 6);
    EXPECT_EQ(network->getNode(NodeId(5)).getNodeType(), Genome::Node::Type::HIDDEN);
    EXPECT_EQ(network->getNumEdges(), 9);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_EQ(out.m_numNodesAdded, 1);
    EXPECT_EQ(out.m_numEdgesAdded, 3);
    EXPECT_TRUE(out.m_newEdges[0].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdges[0].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdges[0].m_newEdge.isValid());
    EXPECT_TRUE(out.m_newEdges[1].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdges[1].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdges[1].m_newEdge.isValid());
    EXPECT_TRUE(out.m_newEdges[2].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdges[2].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdges[2].m_newEdge.isValid());

    // Reset parameter so that no mutation should happen
    params.m_addEdgeMutationRate = 0.0f;
    params.m_addNodeMutationRate = 0.0f;

    genome.mutate(params, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(network->getNumNodes(), 6);
    EXPECT_EQ(network->getNumEdges(), 9);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
    EXPECT_EQ(out.m_numNodesAdded, 0);
    EXPECT_EQ(out.m_numEdgesAdded, 0);
    EXPECT_FALSE(out.m_newEdges[0].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdges[0].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdges[0].m_newEdge.isValid());
    EXPECT_FALSE(out.m_newEdges[1].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdges[1].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdges[1].m_newEdge.isValid());
    EXPECT_FALSE(out.m_newEdges[2].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdges[2].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdges[2].m_newEdge.isValid());

    // Mutate only edge weights
    {
        params.m_weightMutationRate = 1.0f;
        params.m_weightMutationNewValRate = 0.0f;
        const float perturbation = 0.1f;
        params.m_weightMutationPerturbation = perturbation;

        // Remember original edge weights
        std::unordered_map<EdgeId, float> originalWeights;
        const Genome::Network::Edges& edges = network->getEdges();
        for (auto& itr : edges)
        {
            originalWeights[itr.first] = network->getWeightRaw(itr.first);
        }

        genome.mutate(params, out);

        EXPECT_TRUE(genome.validate());

        for (auto& itr : edges)
        {
            if (network->isEdgeEnabled(itr.first))
            {
                float original = originalWeights.at(itr.first);
                float weight = network->getWeightRaw(itr.first);
                EXPECT_TRUE((original * weight) > 0); // Check weight hasn't changed its sign.
                original = std::abs(original);
                weight = std::abs(weight);
                EXPECT_TRUE(weight >= (original * (1.f - perturbation)) && weight <= (original * (1.f + perturbation)));
            }
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

        EXPECT_TRUE(genome.validate());

        const Genome::Network::Edges& edges = network->getEdges();
        for (auto& itr : edges)
        {
            if (network->isEdgeEnabled(itr.first))
            {
                EXPECT_EQ(network->getWeightRaw(itr.first), 3.f);
            }
        }
    }
}

TEST(Genom, CrossOver)
{
    using namespace NEAT;

    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;

    // Create two genomes.
    Genome genome1(cinfo);
    // We reset the counter once here so that genome1 and genome2 have the same initial innovations.
    innovCounter.reset();
    Genome genome2(cinfo);

    float initialEdgeWeightsGenome1[4];
    float initialEdgeWeightsGenome2[4];
    {
        int count = 0;
        for (auto& itr : genome1.getNetwork()->getEdges())
        {
            float weight1 = (float)count;
            genome1.setEdgeWeight(itr.first, weight1);
            initialEdgeWeightsGenome1[count] = weight1;

            float weight2 = (float)(count + 4);
            genome2.setEdgeWeight(itr.first, weight2);
            initialEdgeWeightsGenome2[count] = weight2;

            count++;
        }
    }

    Genome::MutationParams mutParams;
    mutParams.m_weightMutationRate = 0.0f;
    mutParams.m_addEdgeMutationRate = 0.0f;
    mutParams.m_addNodeMutationRate = 1.0f;

    Genome::MutationOut mutOut;

    genome1.mutate(mutParams, mutOut);
    EXPECT_EQ(mutOut.m_numNodesAdded, 1);
    EXPECT_EQ(mutOut.m_numEdgesAdded, 2);
    genome1.mutate(mutParams, mutOut);
    EXPECT_EQ(mutOut.m_numNodesAdded, 1);
    EXPECT_EQ(mutOut.m_numEdgesAdded, 2);
    mutParams.m_addEdgeMutationRate = 1.0f;
    genome1.mutate(mutParams, mutOut);
    EXPECT_EQ(mutOut.m_numNodesAdded, 1);
    EXPECT_EQ(mutOut.m_numEdgesAdded, 3);

    EXPECT_TRUE(genome1.validate());
    EXPECT_EQ(genome1.getNetwork()->getNumNodes(), 7);
    EXPECT_EQ(genome1.getNetwork()->getNumEdges(), 11);

    mutParams.m_addEdgeMutationRate = 0.0f;
    genome2.mutate(mutParams, mutOut);
    EXPECT_EQ(mutOut.m_numNodesAdded, 1);
    EXPECT_EQ(mutOut.m_numEdgesAdded, 2);

    EXPECT_TRUE(genome2.validate());
    EXPECT_EQ(genome2.getNetwork()->getNumNodes(), 5);
    EXPECT_EQ(genome2.getNetwork()->getNumEdges(), 6);

    const EdgeId disabledEdge = mutOut.m_newEdges[0].m_newEdge;
    const_cast<Genome::Network*>(genome2.getNetwork())->setEdgeEnabled(disabledEdge, false);

    Genome::CrossOverParams coParams;
    coParams.m_matchingEdgeSelectionRate = 1.0f;

    Genome newGenome1 = Genome::crossOver(genome1, genome2, false, coParams);

    EXPECT_TRUE(newGenome1.validate());
    EXPECT_EQ(newGenome1.getNetwork()->getNumNodes(), genome1.getNetwork()->getNumNodes());
    EXPECT_EQ(newGenome1.getNetwork()->getNumEdges(), genome1.getNetwork()->getNumEdges());
    for (int i = 0; i < 4; i ++)
    {
        EXPECT_EQ(newGenome1.getNetwork()->getWeightRaw(EdgeId(i)), initialEdgeWeightsGenome1[i]);
    }

    coParams.m_disablingEdgeRate = 1.0f;
    Genome newGenome2 = Genome::crossOver(genome2, genome1, false, coParams);

    EXPECT_TRUE(newGenome2.validate());
    EXPECT_EQ(newGenome2.getNetwork()->getNumNodes(), genome2.getNetwork()->getNumNodes());
    EXPECT_EQ(newGenome2.getNetwork()->getNumEdges(), genome2.getNetwork()->getNumEdges());
    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(newGenome2.getNetwork()->getWeightRaw(EdgeId(i)), initialEdgeWeightsGenome2[i]);
    }
    EXPECT_FALSE(newGenome2.getNetwork()->isEdgeEnabled(disabledEdge));

    coParams.m_matchingEdgeSelectionRate = 0.0f;
    coParams.m_disablingEdgeRate = 0.0f;
    Genome newGenome3 = Genome::crossOver(genome1, genome2, true, coParams);

    EXPECT_TRUE(newGenome3.validate());
    EXPECT_EQ(newGenome3.getNetwork()->getNumNodes(), 8);
    EXPECT_EQ(newGenome3.getNetwork()->getNumEdges(), 13);
    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(newGenome3.getNetwork()->getWeightRaw(EdgeId(i)), initialEdgeWeightsGenome2[i]);
    }
    EXPECT_TRUE(newGenome3.getNetwork()->isEdgeEnabled(disabledEdge));
}
