/*
* DefaultCrossOverTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/GeneticAlgorithms/NEAT/Generators/DefaultCrossOver.h>
#include <NEAT/GeneticAlgorithms/NEAT/Generators/DefaultMutation.h>
#include <NEAT/GeneticAlgorithms/Base/Selectors/GenomeSelector.h>

TEST(DefaultCrossOver, GenerateSingleGenome)
{
    using namespace NEAT;

    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;

    // Create two genomes.
    Genome genome1(cinfo);
    Genome genome2(genome1);

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

    // Mutate genomes several times first
    DefaultMutation mutator;
    mutator.m_params.m_weightMutationRate = 0.0f;
    mutator.m_params.m_addEdgeMutationRate = 0.0f;
    mutator.m_params.m_addNodeMutationRate = 1.0f;

    DefaultMutation::MutationOut mutOut;

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

    mutator.m_params.m_addEdgeMutationRate = 0.0f;
    mutator.mutate(&genome2, mutOut);
    EXPECT_EQ(mutOut.m_numNodesAdded, 1);
    EXPECT_EQ(mutOut.m_numEdgesAdded, 2);

    EXPECT_TRUE(genome2.validate());
    EXPECT_EQ(genome2.getNetwork()->getNumNodes(), 5);
    EXPECT_EQ(genome2.getNetwork()->getNumEdges(), 6);

    const EdgeId disabledEdge = mutOut.m_newEdges[0].m_newEdge;
    const_cast<Genome::Network*>(genome2.getNetwork())->setEdgeEnabled(disabledEdge, false);

    DefaultCrossOver crossOver;
    crossOver.m_params.m_matchingEdgeSelectionRate = 1.0f;

    using GenomePtr = std::shared_ptr<Genome>;

    GenomePtr newGenome1 = std::static_pointer_cast<Genome>(crossOver.crossOver(genome1, genome2, false));

    EXPECT_TRUE(newGenome1->validate());
    EXPECT_EQ(newGenome1->getInputNodes().size(), 2);
    EXPECT_EQ(newGenome1->getNetwork()->getNumNodes(), genome1.getNetwork()->getNumNodes());
    EXPECT_EQ(newGenome1->getNetwork()->getNumEdges(), genome1.getNetwork()->getNumEdges());
    for (int i = 0; i < 4; i++)
    {
        EXPECT_EQ(newGenome1->getNetwork()->getWeightRaw(EdgeId(i)), initialEdgeWeightsGenome1[i]);
    }

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

TEST(DefaultCrossOver, GenerateGeneration)
{
    using namespace NEAT;

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

    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;

    // Create two genomes.
    using GenomePtr = std::shared_ptr<Genome>;
    GenomePtr genome1 = std::make_shared<Genome>(cinfo);
    GenomePtr genome2 = std::make_shared<Genome>(*genome1);
    GenomePtr genome3 = std::make_shared<Genome>(*genome1);
    GenomePtr genome4 = std::make_shared<Genome>(*genome1);

    // Mutate genomes several times first
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

    // Create an array of GenomeData
    using GenomeData = GenerationBase::GenomeData;
    std::vector<GenomeData> genomes;
    genomes.push_back(GenomeData(genome1, GenomeId(0)));
    genomes.push_back(GenomeData(genome2, GenomeId(1)));
    genomes.push_back(GenomeData(genome3, GenomeId(2)));
    genomes.push_back(GenomeData(genome4, GenomeId(3)));

    genomes[1].setFitness(1.0f);
    genomes[2].setFitness(1.0f);

    MyGenomeSelector selector(genomes);

    DefaultCrossOver crossOver;
    crossOver.generate(2, 2, &selector);

    EXPECT_EQ(crossOver.getNumGeneratedGenomes(), 2);
    EXPECT_EQ(crossOver.getGeneratedGenomes()[0]->getNetwork()->getNumEdges(), genome2->getNetwork()->getNumEdges());
    EXPECT_EQ(crossOver.getGeneratedGenomes()[1]->getNetwork()->getNumEdges(), genome3->getNetwork()->getNumEdges());
}
