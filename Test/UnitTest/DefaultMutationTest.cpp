/*
* DefaultMutationTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/GeneticAlgorithms/NEAT/Modifiers/DefaultMutation.h>
#include <UnitTest/Util/TestUtils.h>

TEST(DefaultMutation, MutateSingleGenome)
{
    using namespace NEAT;

    // Create a genome.
    InnovationCounter innovCounter;
    Activation activation = [](float value) { return value * 2.f; };
    activation.m_name = "MyActivation";
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome genome(cinfo);

    const Genome::Network* network = genome.getNetwork();

    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(network->getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

    // All the weights should be 1.0
    {
        const auto& edges = network->getEdges();
        for (const auto& edge : edges)
        {
            EXPECT_EQ(network->getWeight(edge.first), 1.0);
        }
    }

    // Let add node/edge mutation happen all the time
    DefaultMutation mutator;
    mutator.m_params.m_weightMutationRate = 0.0f;
    mutator.m_params.m_addEdgeMutationRate = 1.0f;
    mutator.m_params.m_addNodeMutationRate = 1.0f;
    mutator.m_params.m_removeEdgeMutationRate = 0.f;
    DefaultActivationProvider dap(activation);
    mutator.m_params.m_activationProvider = &dap;

    DefaultMutation::MutationOut out;

    // Mutate the genome.
    // Edges are full connected already so we shouldn't be able to add new edge.
    // A new node should be added and as a result the number of edge should be increased by 2 too.
    mutator.mutate(&genome, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_TRUE(out.m_newNodeInfo.m_nodeId.isValid());
    EXPECT_TRUE(out.m_newNodeInfo.m_previousEdgeId.isValid());
    EXPECT_TRUE(out.m_newNodeInfo.m_newIncomingEdgeId.isValid());
    EXPECT_TRUE(out.m_newNodeInfo.m_newOutgoingEdgeId.isValid());
    EXPECT_EQ(out.m_numEdgesAdded, 2);
    EXPECT_TRUE(out.m_newEdgeInfos[0].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[0].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[0].m_edgeId.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[1].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[1].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[1].m_edgeId.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[2].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[2].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[2].m_edgeId.isValid());
    EXPECT_EQ(network->getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 5);
    EXPECT_EQ(network->getNode(out.m_newNodeInfo.m_nodeId).getNodeType(), Genome::Node::Type::HIDDEN);
    EXPECT_EQ(network->getNumEdges(), 6);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

    // Mutate the genome again.
    // Now we should be able to add both new node and edge.
    // So the number of nodes is +1 and the number of edges is +3
    mutator.mutate(&genome, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_TRUE(out.m_newNodeInfo.m_nodeId.isValid());
    EXPECT_TRUE(out.m_newNodeInfo.m_previousEdgeId.isValid());
    EXPECT_TRUE(out.m_newNodeInfo.m_newIncomingEdgeId.isValid());
    EXPECT_TRUE(out.m_newNodeInfo.m_newOutgoingEdgeId.isValid());
    EXPECT_EQ(out.m_numEdgesAdded, 3);
    EXPECT_TRUE(out.m_newEdgeInfos[0].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[0].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[0].m_edgeId.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[1].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[1].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[1].m_edgeId.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[2].m_sourceInNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[2].m_sourceOutNode.isValid());
    EXPECT_TRUE(out.m_newEdgeInfos[2].m_edgeId.isValid());
    EXPECT_EQ(network->getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 6);
    EXPECT_EQ(network->getNode(out.m_newNodeInfo.m_nodeId).getNodeType(), Genome::Node::Type::HIDDEN);
    std::string activationName("MyActivation");
    EXPECT_EQ(activationName.compare(network->getNode(out.m_newNodeInfo.m_nodeId).getActivationName()), 0);
    EXPECT_EQ(network->getNumEdges(), 9);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

    // Reset parameter so that no mutation should happen
    mutator.m_params.m_addEdgeMutationRate = 0.0f;
    mutator.m_params.m_addNodeMutationRate = 0.0f;

    mutator.mutate(&genome, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_FALSE(out.m_newNodeInfo.m_nodeId.isValid());
    EXPECT_FALSE(out.m_newNodeInfo.m_previousEdgeId.isValid());
    EXPECT_FALSE(out.m_newNodeInfo.m_newIncomingEdgeId.isValid());
    EXPECT_FALSE(out.m_newNodeInfo.m_newOutgoingEdgeId.isValid());
    EXPECT_EQ(out.m_numEdgesAdded, 0);
    EXPECT_FALSE(out.m_newEdgeInfos[0].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[0].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[0].m_edgeId.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[1].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[1].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[1].m_edgeId.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[2].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[2].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[2].m_edgeId.isValid());
    EXPECT_EQ(network->getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 6);
    EXPECT_EQ(network->getNumEdges(), 9);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

    // Mutate only edge weights
    {
        mutator.m_params.m_weightMutationRate = 1.0f;
        mutator.m_params.m_weightMutationNewValRate = 0.0f;
        const float perturbation = 0.1f;
        mutator.m_params.m_weightMutationPerturbation = perturbation;

        // Remember original edge weights
        std::unordered_map<EdgeId, float> originalWeights;
        const auto& edges = network->getEdges();
        for (const auto& edge : edges)
        {
            originalWeights.insert({ edge.first, genome.getEdgeWeightRaw(edge.first) });
        }

        mutator.mutate(&genome, out);

        EXPECT_TRUE(genome.validate());

        // Check the edge mutation was done expectedly.
        for (const auto& edge : edges)
        {
            EdgeId id = edge.first;
            if (genome.isEdgeEnabled(id))
            {
                float original = originalWeights.at(id);
                float weight = genome.getEdgeWeightRaw(id);
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
        mutator.m_params.m_weightMutationNewValRate = 1.0f;
        mutator.m_params.m_random = &random;

        mutator.mutate(&genome, out);

        EXPECT_TRUE(genome.validate());

        const auto& edges = network->getEdges();
        for (const auto& edge : edges)
        {
            EdgeId id = edge.first;
            if (genome.isEdgeEnabled(id))
            {
                EXPECT_EQ(genome.getEdgeWeightRaw(id), 3.f);
            }
        }
    }

    // Reset parameter so that only edge removal can happen.
    mutator.m_params.m_addEdgeMutationRate = 0.0f;
    mutator.m_params.m_addNodeMutationRate = 0.0f;
    mutator.m_params.m_weightMutationNewValRate = 0.0f;
    mutator.m_params.m_weightMutationRate = 0.0f;
    mutator.m_params.m_removeEdgeMutationRate = 1.0f;

    mutator.mutate(&genome, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_FALSE(out.m_newNodeInfo.m_nodeId.isValid());
    EXPECT_FALSE(out.m_newNodeInfo.m_previousEdgeId.isValid());
    EXPECT_FALSE(out.m_newNodeInfo.m_newIncomingEdgeId.isValid());
    EXPECT_FALSE(out.m_newNodeInfo.m_newOutgoingEdgeId.isValid());
    EXPECT_EQ(out.m_numEdgesAdded, 0);
    EXPECT_FALSE(out.m_newEdgeInfos[0].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[0].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[0].m_edgeId.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[1].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[1].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[1].m_edgeId.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[2].m_sourceInNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[2].m_sourceOutNode.isValid());
    EXPECT_FALSE(out.m_newEdgeInfos[2].m_edgeId.isValid());
    EXPECT_EQ(network->getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 6);
    EXPECT_EQ(network->getNumEdges(), 8);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

}

TEST(DefaultMutation, MutateGeneration)
{
    using namespace NEAT;
    using namespace TestUtils;
    using GenomePtr = std::shared_ptr<Genome>;

    // Custom random generator which always selects the minimum integer.
    class MyRandom : public PseudoRandom
    {
    public:
        MyRandom() : PseudoRandom(0) {}
        virtual int randomInteger(int min, int max) override { return min; }
    };

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    Activation activation = [](float value) { return value * 2.f; };
    activation.m_name = "MyActivation";
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    GenomePtr genome1 = std::make_shared<Genome>(cinfo);

    // Verify the initial condition of the genome.
    {
        const Genome::Network* network = genome1->getNetwork();

        EXPECT_TRUE(genome1->validate());
        EXPECT_EQ(genome1->getInputNodes().size(), 2);
        EXPECT_EQ(genome1->getNumNodes(), 4);
        EXPECT_EQ(genome1->getNumEdges(), 4);
        EXPECT_EQ(genome1->getOutputNodes().size(), 2);

        // All the weights should be 1.0
        {
            const auto& edges = network->getEdges();
            for (const auto& edge : edges)
            {
                EXPECT_EQ(genome1->getEdgeWeight(edge.first), 1.0);
            }
        }
    }

    // Let add node/edge mutation happen all the time.
    MyRandom random;
    DefaultMutation mutator;
    mutator.m_params.m_weightMutationRate = 0.0f;
    mutator.m_params.m_addEdgeMutationRate = 1.0f;
    mutator.m_params.m_addNodeMutationRate = 1.0f;
    mutator.m_params.m_removeEdgeMutationRate = 0.f;
    mutator.m_params.m_random = &random;

    {
        // Mutate the genome once in order to make the genome1 not fully connected.
        DefaultMutation::MutationOut out;
        mutator.mutate(genome1.get(), out);

        EXPECT_TRUE(genome1->validate());
        EXPECT_TRUE(out.m_newNodeInfo.m_nodeId.isValid());
        EXPECT_TRUE(out.m_newNodeInfo.m_previousEdgeId.isValid());
        EXPECT_TRUE(out.m_newNodeInfo.m_newIncomingEdgeId.isValid());
        EXPECT_TRUE(out.m_newNodeInfo.m_newOutgoingEdgeId.isValid());
        EXPECT_EQ(out.m_numEdgesAdded, 2);
        EXPECT_TRUE(out.m_newEdgeInfos[0].m_sourceInNode.isValid());
        EXPECT_TRUE(out.m_newEdgeInfos[0].m_sourceOutNode.isValid());
        EXPECT_TRUE(out.m_newEdgeInfos[0].m_edgeId.isValid());
        EXPECT_TRUE(out.m_newEdgeInfos[1].m_sourceInNode.isValid());
        EXPECT_TRUE(out.m_newEdgeInfos[1].m_sourceOutNode.isValid());
        EXPECT_TRUE(out.m_newEdgeInfos[1].m_edgeId.isValid());
        EXPECT_FALSE(out.m_newEdgeInfos[2].m_sourceInNode.isValid());
        EXPECT_FALSE(out.m_newEdgeInfos[2].m_sourceOutNode.isValid());
        EXPECT_FALSE(out.m_newEdgeInfos[2].m_edgeId.isValid());
        EXPECT_EQ(genome1->getInputNodes().size(), 2);
        EXPECT_EQ(genome1->getNumNodes(), 5);
        EXPECT_EQ(genome1->getNetwork()->getNode(out.m_newNodeInfo.m_nodeId).getNodeType(), Genome::Node::Type::HIDDEN);
        EXPECT_EQ(genome1->getNumEdges(), 6);
        EXPECT_EQ(genome1->getOutputNodes().size(), 2);
    }

    // Create one more genome by copying genome1.
    GenomePtr genome2 = std::make_shared<Genome>(*genome1);
    EXPECT_TRUE(genome2->validate());
    EXPECT_TRUE(compareGenomeWithWeightsAndStates(*genome1, *genome2));

    // Modify two new genomes by mutation.
    mutator.modifyGenomes(std::static_pointer_cast<GenomeBase>(genome1));
    mutator.modifyGenomes(std::static_pointer_cast<GenomeBase>(genome2));

    // Create an array of GenomeData.
    std::vector<GenomePtr> genomes;
    genomes.push_back(genome1);
    genomes.push_back(genome2);

    // The exact same mutation should have happened for both descendants of genome1 and genome2.
    // Newly added edges in genome1 and genome2 are the same location, so they should have assigned the same innovation ids.
    for (const auto& g : genomes)
    {
        const Genome* newGenome = static_cast<const Genome*>(g.get());
        EXPECT_TRUE(newGenome->validate());
        EXPECT_EQ(newGenome->getInputNodes().size(), 2);
        EXPECT_EQ(newGenome->getNumNodes(), 6);
        EXPECT_EQ(newGenome->getNumEdges(), 9);
        EXPECT_EQ(newGenome->getOutputNodes().size(), 2);
    }

    // Compare the two modified genomes.
    // The two new genomes should have the identical structure but edge weights should be different.
    {
        EXPECT_TRUE(compareGenome(*genome1, *genome2));
        EXPECT_FALSE(compareGenomeWithWeightsAndStates(*genome1, *genome2));
    }
}
