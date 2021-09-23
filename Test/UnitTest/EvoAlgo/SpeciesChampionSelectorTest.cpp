/*
* GenomeTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <EvoAlgo/GeneticAlgorithms/NEAT/Generators/SpeciesChampionSelector.h>
#include <EvoAlgo/GeneticAlgorithms/NEAT/Modifiers/DefaultMutation.h>

TEST(SpeciesChampionSelector, SelectChampions)
{
    using namespace NEAT;
    using GenomePtr = std::shared_ptr<Genome>;
    using SpeciesPtr = SpeciesChampionSelector::SpeciesPtr;
    using SpeciesList = SpeciesChampionSelector::SpeciesList;

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    {
        cinfo.m_numInputNodes = 2;
        cinfo.m_numOutputNodes = 2;
        cinfo.m_innovIdCounter = &innovCounter;
    }
    GenomePtr initGenome = std::make_shared<Genome>(cinfo);

    // Create a genome to add the species
    GenomePtr genome1 = std::make_shared<Genome>(*initGenome.get());

    // Mutate the genome
    DefaultMutation::MutationParams mutParams;
    mutParams.m_addEdgeMutationRate = 1.0f;
    mutParams.m_addNodeMutationRate = 1.0f;
    DefaultMutation mutator(mutParams);
    DefaultMutation::MutationOut mutOut;
    mutator.mutate(genome1.get(), mutOut);

    // Create one more genome to add the species
    GenomePtr genome2 = std::make_shared<Genome>(*genome1);

    // Mutate the genome
    mutator.mutate(genome2.get(), mutOut);

    // Set up calc distance params.
    Genome::CalcDistParams calcDistParams;
    {
        calcDistParams.m_disjointFactor = 1.0f;
        calcDistParams.m_weightFactor = 1.0f;
    }

    // Create three species.
    SpeciesPtr s0 = std::make_shared<Species>(initGenome, 1.0f);
    SpeciesPtr s1 = std::make_shared<Species>(initGenome, 2.0f);
    s1->tryAddGenome(genome1, 3.0f, 100.0f, calcDistParams);
    SpeciesPtr s2 = std::make_shared<Species>(initGenome, 4.0f);
    s2->tryAddGenome(genome1, 5.0f, 100.0f, calcDistParams);
    s2->tryAddGenome(genome2, 6.0f, 100.0f, calcDistParams);

    SpeciesList species;
    species.insert({ SpeciesId(0), s0 });
    species.insert({ SpeciesId(1), s1 });
    species.insert({ SpeciesId(2), s2 });

    SpeciesChampionSelector scs(2);

    // Try to generate genomes without setting species.
    scs.generate(6, 2, nullptr);
    EXPECT_EQ(scs.getNumGeneratedGenomes(), 0);

    // Set species.
    scs.updateSpecies(species);

    // Generate no genome.
    scs.generate(6, 0, nullptr);
    EXPECT_EQ(scs.getNumGeneratedGenomes(), 0);

    // Generate genomes.
    scs.generate(6, 3, nullptr);
    EXPECT_EQ(scs.getNumGeneratedGenomes(), 2);
    EXPECT_EQ(scs.getGeneratedGenomes()[0]->getNumEdges(), genome1->getNumEdges());
    EXPECT_EQ(scs.getGeneratedGenomes()[1]->getNumEdges(), genome2->getNumEdges());

    // Make s2 not reproducible.
    s2->setReproducible(false);
    scs.generate(6, 3, nullptr);
    EXPECT_EQ(scs.getNumGeneratedGenomes(), 1);
    EXPECT_EQ(scs.getGeneratedGenomes()[0]->getNumEdges(), genome1->getNumEdges());
    s2->setReproducible(true);

    // Generate only one genome.
    scs.generate(6, 1, nullptr);
    EXPECT_EQ(scs.getNumGeneratedGenomes(), 1);
    EXPECT_EQ(scs.getGeneratedGenomes()[0]->getNumEdges(), genome2->getNumEdges());

}
