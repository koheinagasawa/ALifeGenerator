/*
* DefaultCrossOverTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/GeneticAlgorithms/NEAT/Generators/DefaultCrossOver.h>
#include <NEAT/GeneticAlgorithms/NEAT/Modifiers/DefaultMutation.h>
#include <NEAT/GeneticAlgorithms/Base/Selectors/GenomeSelector.h>

TEST(DefaultCrossOver, GenerateSingleGenome)
{
    using namespace NEAT;
    using GenomePtr = std::shared_ptr<Genome>;

    // Create two genomes.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    Genome genome1(cinfo);
    Genome genome2(genome1);

    // Set the initial edge weights and store the values.
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

    // Mutate genomes several times first.
    DefaultMutation::MutationOut mutOut;
    {
        DefaultMutation mutator;
        mutator.m_params.m_weightMutationRate = 0.0f;
        mutator.m_params.m_addEdgeMutationRate = 0.0f;
        mutator.m_params.m_addNodeMutationRate = 1.0f;

        // Mutate genome1 three times.
        mutator.mutate(&genome1, mutOut);
        EXPECT_EQ(mutOut.m_numNodesAdded, 1);
        EXPECT_EQ(mutOut.m_numEdgesAdded, 2);
        mutator.mutate(&genome1, mutOut);
        EXPECT_EQ(mutOut.m_numNodesAdded, 1);
        EXPECT_EQ(mutOut.m_numEdgesAdded, 2);
        mutator.m_params.m_addEdgeMutationRate = 1.0f;
        mutator.mutate(&genome1, mutOut);
        EXPECT_EQ(mutOut.m_numNodesAdded, 1);
        EXPECT_EQ(mutOut.m_numEdgesAdded, 3);

        EXPECT_TRUE(genome1.validate());
        EXPECT_EQ(genome1.getNetwork()->getNumNodes(), 7);
        EXPECT_EQ(genome1.getNetwork()->getNumEdges(), 11);

        // Mutate genome2 once.
        mutator.m_params.m_addEdgeMutationRate = 0.0f;
        mutator.mutate(&genome2, mutOut);
        EXPECT_EQ(mutOut.m_numNodesAdded, 1);
        EXPECT_EQ(mutOut.m_numEdgesAdded, 2);

        EXPECT_TRUE(genome2.validate());
        EXPECT_EQ(genome2.getNetwork()->getNumNodes(), 5);
        EXPECT_EQ(genome2.getNetwork()->getNumEdges(), 6);
    }

    // Disable one edge in genome2.
    const EdgeId disabledEdge = mutOut.m_newEdges[0].m_newEdge;
    const_cast<Genome::Network*>(genome2.getNetwork())->setEdgeEnabled(disabledEdge, false);

    // Set up cross over.
    DefaultCrossOver crossOver;
    crossOver.m_params.m_matchingEdgeSelectionRate = 1.0f;

    // Generate a genome by cross over using genome1 as a better offspring.
    {
        GenomePtr newGenome1 = std::static_pointer_cast<Genome>(crossOver.crossOver(genome1, genome2, false));

        EXPECT_TRUE(newGenome1->validate());
        EXPECT_EQ(newGenome1->getInputNodes().size(), 2);
        EXPECT_EQ(newGenome1->getNetwork()->getNumNodes(), genome1.getNetwork()->getNumNodes());
        EXPECT_EQ(newGenome1->getNetwork()->getNumEdges(), genome1.getNetwork()->getNumEdges());
        for (int i = 0; i < 4; i++)
        {
            EXPECT_EQ(newGenome1->getNetwork()->getWeightRaw(EdgeId(i)), initialEdgeWeightsGenome1[i]);
        }
    }

    // Generate a genome by cross over using genome2 as a better offspring.
    // In this setting, disabled edges in either parent should become disabled too.
    {
        crossOver.m_params.m_disablingEdgeRate = 1.0f;
        GenomePtr newGenome2 = std::static_pointer_cast<Genome>(crossOver.crossOver(genome2, genome1, false));

        EXPECT_TRUE(newGenome2->validate());
        EXPECT_EQ(newGenome2->getInputNodes().size(), 2);
        EXPECT_EQ(newGenome2->getNetwork()->getNumNodes(), genome2.getNetwork()->getNumNodes());
        EXPECT_EQ(newGenome2->getNetwork()->getNumEdges(), genome2.getNetwork()->getNumEdges());
        for (int i = 0; i < 4; i++)
        {
            EXPECT_EQ(newGenome2->getNetwork()->getWeightRaw(EdgeId(i)), initialEdgeWeightsGenome2[i]);
        }
        EXPECT_FALSE(newGenome2->getNetwork()->isEdgeEnabled(disabledEdge));
    }

    // Generate a genome by cross over using genome1 as a better offspring.
    // In this setting, weights of matching edges are inherited from the secondary genome, which is genome2 here.
    {
        crossOver.m_params.m_matchingEdgeSelectionRate = 0.0f;
        crossOver.m_params.m_disablingEdgeRate = 0.0f;
        GenomePtr newGenome3 = std::static_pointer_cast<Genome>(crossOver.crossOver(genome1, genome2, true));

        EXPECT_TRUE(newGenome3->validate());
        EXPECT_EQ(newGenome3->getInputNodes().size(), 2);
        EXPECT_EQ(newGenome3->getNetwork()->getNumNodes(), 8);
        EXPECT_EQ(newGenome3->getNetwork()->getNumEdges(), 13);
        for (int i = 0; i < 4; i++)
        {
            EXPECT_EQ(newGenome3->getNetwork()->getWeightRaw(EdgeId(i)), initialEdgeWeightsGenome2[i]);
        }
        EXPECT_TRUE(newGenome3->getNetwork()->isEdgeEnabled(disabledEdge));
    }
}

TEST(DefaultCrossOver, GenerateGeneration)
{
    using namespace NEAT;
    using GenomePtr = std::shared_ptr<Genome>;
    using GenomeData = GenerationBase::GenomeData;

    // Custom selector class to select genome incrementally.
    class MyGenomeSelector : public GenomeSelector
    {
    public:
        MyGenomeSelector(const GenomeDatas& genomes) : m_genomes(genomes) {}
        virtual auto selectGenome()->const GenomeData* override { assert(0); return nullptr; }
        virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override
        {
            genome1 = &m_genomes[m_index++];
            genome2 = &m_genomes[m_index++];
        }
    protected:
        const GenomeDatas& m_genomes;
        int m_index = 0;
    };

    // Create four genomes.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    GenomePtr genome1 = std::make_shared<Genome>(cinfo);
    GenomePtr genome2 = std::make_shared<Genome>(*genome1);
    GenomePtr genome3 = std::make_shared<Genome>(*genome1);
    GenomePtr genome4 = std::make_shared<Genome>(*genome1);

    // Mutate genomes several times first.
    {
        DefaultMutation mutator;
        mutator.m_params.m_weightMutationRate = 1.0f;
        mutator.m_params.m_addEdgeMutationRate = 1.0f;
        mutator.m_params.m_addNodeMutationRate = 1.0f;

        DefaultMutation::MutationOut mutateOut;
        mutator.mutate(genome2.get(), mutateOut);
        mutator.mutate(genome3.get(), mutateOut);
        mutator.mutate(genome3.get(), mutateOut);
        mutator.mutate(genome4.get(), mutateOut);
        mutator.mutate(genome4.get(), mutateOut);
        mutator.mutate(genome4.get(), mutateOut);
    }

    // Create an array of GenomeData.
    std::vector<GenomeData> genomes;
    {
        genomes.push_back(GenomeData(genome1, GenomeId(0)));
        genomes.push_back(GenomeData(genome2, GenomeId(1)));
        genomes.push_back(GenomeData(genome3, GenomeId(2)));
        genomes.push_back(GenomeData(genome4, GenomeId(3)));

        // Set genomes' fitness.
        genomes[1].setFitness(1.0f);
        genomes[2].setFitness(1.0f);
    }

    // Create a custom genome selector.
    MyGenomeSelector selector(genomes);

    // Create a cross over delegate.
    DefaultCrossOver crossOver;

    // Generate no genome.
    crossOver.generate(2, 0, &selector);
    EXPECT_EQ(crossOver.getNumGeneratedGenomes(), 0);

    // Generate two genomes.
    // By the custom selector, genome1-genome2 pair and genome3-genome4 pair will be cross-over-ed.
    // genome2 and genome3 have better fitness.
    crossOver.generate(2, 2, &selector);
    EXPECT_EQ(crossOver.getNumGeneratedGenomes(), 2);
    EXPECT_EQ(crossOver.getGeneratedGenomes()[0]->getNetwork()->getNumEdges(), genome2->getNetwork()->getNumEdges());
    EXPECT_EQ(crossOver.getGeneratedGenomes()[1]->getNetwork()->getNumEdges(), genome3->getNetwork()->getNumEdges());
}
