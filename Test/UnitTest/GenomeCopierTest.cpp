/*
* GenomeCopierTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/GeneticAlgorithms/Base/Generators/GenomeCopier.h>
#include <NEAT/GeneticAlgorithms/Base/Selectors/GenomeSelector.h>
#include <NEAT/GeneticAlgorithms/NEAT/Genome.h>
#include <NEAT/GeneticAlgorithms/NEAT/Modifiers/DefaultMutation.h>

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

TEST(GenomeCopier, CopyGenome)
{
    using namespace NEAT;
    using GenomePtr = std::shared_ptr<Genome>;
    using GenomeData = GenerationBase::GenomeData;

    // Create a copier
    GenomeCopier<GenomeBase> copier;

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
    copier.generate(3, 3, &selector);
}
