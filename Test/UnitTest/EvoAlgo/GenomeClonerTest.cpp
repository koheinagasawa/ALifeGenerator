/*
* GenomeClonerTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <EvoAlgo/GeneticAlgorithms/Base/Generators/GenomeCloner.h>
#include <EvoAlgo/GeneticAlgorithms/Base/Selectors/GenomeSelector.h>
#include <EvoAlgo/GeneticAlgorithms/NEAT/Genome.h>
#include <EvoAlgo/GeneticAlgorithms/NEAT/Modifiers/DefaultMutation.h>
#include <UnitTest/Util/TestUtils.h>

namespace
{
    class MyGenomeSelector : public GenomeSelector
    {
    public:
        MyGenomeSelector(const GenomeDatas& genomes) : m_genomes(genomes) {}
        virtual auto selectGenome()->const GenomeData* override { return &m_genomes[m_index++]; }
        virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override { assert(0); }

        const GenomeDatas& m_genomes;
        int m_index = 0;
    };
}

TEST(GenomeCloner, CopyGenome)
{
    using namespace NEAT;
    using namespace TestUtils;
    using GenomePtr = std::shared_ptr<Genome>;
    using GenomeData = GenerationBase::GenomeData;

    // Create a cloner
    GenomeCloner<GenomeBase> cloner;

    // Create three genomes.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    GenomePtr genome1 = std::make_shared<Genome>(cinfo);
    GenomePtr genome2 = std::make_shared<Genome>(*genome1);
    GenomePtr genome3 = std::make_shared<Genome>(*genome1);

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
    }

    // Create an array of GenomeData.
    std::vector<GenomeData> genomes;
    {
        genomes.push_back(GenomeData(genome1, GenomeId(0)));
        genomes.push_back(GenomeData(genome2, GenomeId(1)));
        genomes.push_back(GenomeData(genome3, GenomeId(2)));
    }

    // Create a selector.
    MyGenomeSelector selector(genomes);

    // Copy
    cloner.generate(3, 3, &selector);

    EXPECT_TRUE(compareGenomeWithWeightsAndStates(*static_cast<Genome*>(cloner.getGeneratedGenomes()[0].get()), *genome1));
    EXPECT_TRUE(compareGenomeWithWeightsAndStates(*static_cast<Genome*>(cloner.getGeneratedGenomes()[1].get()), *genome2));
    EXPECT_TRUE(compareGenomeWithWeightsAndStates(*static_cast<Genome*>(cloner.getGeneratedGenomes()[2].get()), *genome3));
}
