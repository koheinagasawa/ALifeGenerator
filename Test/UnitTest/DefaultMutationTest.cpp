/*
* DefaultMutationTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/DefaultMutation.h>
#include <NEAT/GenomeSelector.h>

namespace
{
    using namespace NEAT;

    bool compareGenome(const Genome& g1, const Genome& g2)
    {
        const Genome::Network* net1 = g1.getNetwork();
        const Genome::Network* net2 = g2.getNetwork();
        if (net1->getNumNodes() != net2->getNumNodes()) return false;
        if (net1->getNumEdges() != net2->getNumEdges()) return false;

        for (auto& itr : net1->getNodes())
        {
            NodeId id = itr.first;
            if (!net2->hasNode(id)) return false;
            const Genome::Node& n1 = net1->getNode(id);
            const Genome::Node& n2 = net2->getNode(id);

            if (n1.getNodeType() != n2.getNodeType()) return false;

            const auto& inEdges1 = net1->getIncomingEdges(id);
            const auto& inEdges2 = net2->getIncomingEdges(id);

            if (inEdges1.size() != inEdges2.size()) return false;

            for (int i = 0; i < (int)inEdges1.size(); i++)
            {
                if (inEdges1[i] != inEdges2[i]) return false;
            }
        }

        for (auto& itr : net1->getEdges())
        {
            EdgeId id = itr.first;
            if (!net2->hasEdge(id)) return false;

            const Genome::Edge& e1 = itr.second;
            const Genome::Edge& e2 = net2->getEdges().at(id);

            if (e1.getInNode() != e2.getInNode()) return false;
            if (e1.getOutNode() != e2.getOutNode()) return false;
        }

        return true;
    }

    bool compareGenomeWithWeightsAndStates(const Genome& g1, const Genome& g2)
    {
        if (!compareGenome(g1, g2)) return false;

        const Genome::Network* net1 = g1.getNetwork();
        const Genome::Network* net2 = g2.getNetwork();

        for (auto& itr : net1->getEdges())
        {
            EdgeId id = itr.first;
            if (!net2->hasEdge(id)) return false;

            const Genome::Edge& e1 = itr.second;
            const Genome::Edge& e2 = net2->getEdges().at(id);

            if (e1.getWeightRaw() != e2.getWeightRaw()) return false;
            if (e1.isEnabled() != e2.isEnabled()) return false;
        }

        return true;
    }
}

TEST(DefaultMutation, MutateSingleGenome)
{
    using namespace NEAT;

    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome::Activation activation = [](float value) { return value * 2.f; };
    activation.m_name = "MyActivation";
    cinfo.m_defaultActivation = &activation;

    // Create a genome.
    Genome genome(cinfo);

    const Genome::Network* network = genome.getNetwork();

    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

    // All the weights should be 1.0
    {
        const Genome::Network::Edges& edges = network->getEdges();
        for (auto& itr : edges)
        {
            EXPECT_EQ(network->getWeight(itr.first), 1.0);
        }
    }

    // Let add node/edge mutation happen all the time
    DefaultMutation mutator;
    mutator.m_params.m_weightMutationRate = 0.0f;
    mutator.m_params.m_addEdgeMutationRate = 1.0f;
    mutator.m_params.m_addNodeMutationRate = 1.0f;

    DefaultMutation::MutationOut out;

    // Mutate the genome.
    // Edges are full connected already so we shouldn't be able to add new edge.
    // A new node should be added and as a result the number of edge should be increased by 2 too.
    mutator.mutate(&genome, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(out.m_numNodesAdded, 1);
    EXPECT_TRUE(out.m_newNode.m_newNode.isValid());
    EXPECT_TRUE(out.m_newNode.m_previousEdgeId.isValid());
    EXPECT_TRUE(out.m_newNode.m_newIncomingEdgeId.isValid());
    EXPECT_TRUE(out.m_newNode.m_newOutgoingEdgeId.isValid());
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
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 5);
    EXPECT_EQ(network->getNode(out.m_newNode.m_newNode).getNodeType(), Genome::Node::Type::HIDDEN);
    EXPECT_EQ(network->getNumEdges(), 6);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

    // Mutate the genome again.
    // Now we should be able to add both new node and edge.
    // So the number of nodes is +1 and the number of edges is +3
    mutator.mutate(&genome, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(out.m_numNodesAdded, 1);
    EXPECT_TRUE(out.m_newNode.m_newNode.isValid());
    EXPECT_TRUE(out.m_newNode.m_previousEdgeId.isValid());
    EXPECT_TRUE(out.m_newNode.m_newIncomingEdgeId.isValid());
    EXPECT_TRUE(out.m_newNode.m_newOutgoingEdgeId.isValid());
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
    EXPECT_EQ(genome.getInputNodes().size(), 2);
    EXPECT_EQ(network->getNumNodes(), 6);
    EXPECT_EQ(network->getNode(out.m_newNode.m_newNode).getNodeType(), Genome::Node::Type::HIDDEN);
    EXPECT_EQ(network->getNode(out.m_newNode.m_newNode).getActivationName(), "MyActivation");
    EXPECT_EQ(network->getNumEdges(), 9);
    EXPECT_EQ(network->getOutputNodes().size(), 2);

    // Reset parameter so that no mutation should happen
    mutator.m_params.m_addEdgeMutationRate = 0.0f;
    mutator.m_params.m_addNodeMutationRate = 0.0f;

    mutator.mutate(&genome, out);

    EXPECT_TRUE(genome.validate());
    EXPECT_EQ(out.m_numNodesAdded, 0);
    EXPECT_FALSE(out.m_newNode.m_newNode.isValid());
    EXPECT_FALSE(out.m_newNode.m_previousEdgeId.isValid());
    EXPECT_FALSE(out.m_newNode.m_newIncomingEdgeId.isValid());
    EXPECT_FALSE(out.m_newNode.m_newOutgoingEdgeId.isValid());
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
    EXPECT_EQ(genome.getInputNodes().size(), 2);
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
        const Genome::Network::Edges& edges = network->getEdges();
        for (auto& itr : edges)
        {
            originalWeights.insert({ itr.first, network->getWeightRaw(itr.first) });
        }

        mutator.mutate(&genome, out);

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
        mutator.m_params.m_weightMutationNewValRate = 1.0f;
        mutator.m_params.m_random = &random;

        mutator.mutate(&genome, out);

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

TEST(DefaultMutation, MutateGeneration)
{
    // Custom selector class to select genome incrementally.
    class MyGenomeSelector : public GenomeSelector
    {
    public:
        virtual bool setGenomes(const GenomeDatas& genomes) { m_genomes = &genomes; m_index = 0; return true; }
        virtual auto selectGenome()->const GenomeData* { return &(*m_genomes)[m_index++]; }
        virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) { assert(0); }
    protected:
        const GenomeDatas* m_genomes;
        int m_index;
    };

    // Custom random generator which always selects the minimum integer.
    class MyRandom : public PseudoRandom
    {
    public:
        MyRandom() : PseudoRandom(0) {}
        virtual int randomInteger(int min, int max) override { return min; }
    };

    using namespace NEAT;

    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome::Activation activation = [](float value) { return value * 2.f; };
    activation.m_name = "MyActivation";
    cinfo.m_defaultActivation = &activation;

    using GenomePtr = std::shared_ptr<Genome>;

    // Create a genome.
    GenomePtr genome1 = std::make_shared<Genome>(cinfo);

    {
        const Genome::Network* network = genome1->getNetwork();

        EXPECT_TRUE(genome1->validate());
        EXPECT_EQ(genome1->getInputNodes().size(), 2);
        EXPECT_EQ(network->getNumNodes(), 4);
        EXPECT_EQ(network->getNumEdges(), 4);
        EXPECT_EQ(network->getOutputNodes().size(), 2);

        // All the weights should be 1.0
        {
            const Genome::Network::Edges& edges = network->getEdges();
            for (auto& itr : edges)
            {
                EXPECT_EQ(network->getWeight(itr.first), 1.0);
            }
        }
    }

    // Let add node/edge mutation happen all the time
    MyRandom random;
    DefaultMutation mutator;
    mutator.m_params.m_weightMutationRate = 0.0f;
    mutator.m_params.m_addEdgeMutationRate = 1.0f;
    mutator.m_params.m_addNodeMutationRate = 1.0f;
    mutator.m_params.m_random = &random;

    {
        // Mutate the genome once in order to make the genome1 not fully connected.
        DefaultMutation::MutationOut out;
        mutator.mutate(genome1.get(), out);

        const Genome::Network* network = genome1->getNetwork();
        EXPECT_TRUE(genome1->validate());
        EXPECT_EQ(out.m_numNodesAdded, 1);
        EXPECT_TRUE(out.m_newNode.m_newNode.isValid());
        EXPECT_TRUE(out.m_newNode.m_previousEdgeId.isValid());
        EXPECT_TRUE(out.m_newNode.m_newIncomingEdgeId.isValid());
        EXPECT_TRUE(out.m_newNode.m_newOutgoingEdgeId.isValid());
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
        EXPECT_EQ(genome1->getInputNodes().size(), 2);
        EXPECT_EQ(network->getNumNodes(), 5);
        EXPECT_EQ(network->getNode(out.m_newNode.m_newNode).getNodeType(), Genome::Node::Type::HIDDEN);
        EXPECT_EQ(network->getNumEdges(), 6);
        EXPECT_EQ(network->getOutputNodes().size(), 2);
    }

    // Copy the genome
    GenomePtr genome2 = std::make_shared<Genome>(*genome1);
    EXPECT_TRUE(genome2->validate());
    EXPECT_TRUE(compareGenomeWithWeightsAndStates(*genome1, *genome2));

    // Create an array of GenomeData
    using GenomeData = GenerationBase::GenomeData;
    std::vector<GenomeData> genomes;
    genomes.push_back(GenomeData(genome1, GenomeId(0)));
    genomes.push_back(GenomeData(genome2, GenomeId(1)));

    // Set up the custom selector.
    MyGenomeSelector selector;
    selector.setGenomes(genomes);

    EXPECT_EQ(mutator.getNumGeneratedGenomes(), 0);
    EXPECT_EQ(mutator.getGeneratedGenomes().size(), 0);

    mutator.m_params.m_mutatedGenomesRate = 1.0f;

    // Generate new genomes by mutation
    mutator.generate(2, 2, &selector);

    // The exact same mutation should have happened for both descendants of genome1 and genome2.
    // Newly added edges in genome1 and genome2 are the same location, so they should have assigned the same innovation ids.
    EXPECT_EQ(mutator.getNumGeneratedGenomes(), 2);
    EXPECT_EQ(mutator.getGeneratedGenomes().size(), 2);
    for (const auto& g : mutator.getGeneratedGenomes())
    {
        const Genome* newGenome = static_cast<const Genome*>(g.get());
        const Genome::Network* network = newGenome->getNetwork();
        EXPECT_TRUE(newGenome->validate());
        EXPECT_EQ(genome1->getInputNodes().size(), 2);
        EXPECT_EQ(network->getNumNodes(), 6);
        EXPECT_EQ(network->getNumEdges(), 9);
        EXPECT_EQ(network->getOutputNodes().size(), 2);
    }

    // Compare the two newly generated genomes.
    {
        const Genome* newGenome1 = static_cast<const Genome*>(mutator.getGeneratedGenomes()[0].get());
        const Genome* newGenome2 = static_cast<const Genome*>(mutator.getGeneratedGenomes()[1].get());
        EXPECT_TRUE(compareGenome(*newGenome1, *newGenome2));
        EXPECT_FALSE(compareGenomeWithWeightsAndStates(*newGenome1, *newGenome2));
    }

    // Change the parameter and call generate again.
    mutator.m_params.m_mutatedGenomesRate = 0.0f;
    mutator.generate(2, 2, &selector);
    EXPECT_EQ(mutator.getNumGeneratedGenomes(), 0);
    EXPECT_EQ(mutator.getGeneratedGenomes().size(), 0);
}
