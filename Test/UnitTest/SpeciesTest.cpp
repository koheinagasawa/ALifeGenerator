/*
* SpeciesTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/GeneticAlgorithms/NEAT/Species.h>
#include <NEAT/GeneticAlgorithms/NEAT/Modifiers/DefaultMutation.h>

TEST(Species, AddGenomeToSpecies)
{
    using namespace NEAT;
    using GenomePtr = std::shared_ptr<Genome>;

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    {
        cinfo.m_numInputNodes = 2;
        cinfo.m_numOutputNodes = 2;
        cinfo.m_innovIdCounter = &innovCounter;
    }
    Genome initGenome(cinfo);

    // Create a species.
    Species species(initGenome);

    EXPECT_EQ(species.getNumMembers(), 0);
    EXPECT_EQ(species.getBestGenome(), nullptr);
    EXPECT_EQ(species.getBestFitness(), 0);

    // Create a genome to add the species.
    GenomePtr genome1 = std::make_shared<Genome>(initGenome);

    // Mutate the genome.
    {
        DefaultMutation::MutationParams mutParams;
        mutParams.m_weightMutationRate = 0.0f;
        mutParams.m_addEdgeMutationRate = 0.0f;
        mutParams.m_addNodeMutationRate = 1.0f;

        DefaultMutation mutator(mutParams);

        DefaultMutation::MutationOut mutOut;
        mutator.mutate(genome1.get(), mutOut);
    }

    // Set up calc distance params.
    Genome::CalcDistParams calcDistParams;
    {
        calcDistParams.m_disjointFactor = 1.0f;
        calcDistParams.m_weightFactor = 1.0f;
    }

    EXPECT_EQ(species.getStagnantGenerationCount(), 0);

    // Prepare for the new generation.
    species.preNewGeneration();

    // Try to add the genome to the species.
    // This should fail to add since distance threshold is too small.
    EXPECT_FALSE(species.tryAddGenome(genome1, 1.f, 0.0001f, calcDistParams));

    EXPECT_EQ(species.getNumMembers(), 0);

    // This should succeed to add.
    EXPECT_TRUE(species.tryAddGenome(genome1, 1.f, 5.f, calcDistParams));

    EXPECT_EQ(species.getNumMembers(), 1);

    // Finalize the generation.
    species.postNewGeneration();

    EXPECT_EQ(species.getBestGenome(), genome1);
    EXPECT_EQ(species.getBestFitness(), 1.f);
    EXPECT_EQ(species.getStagnantGenerationCount(), 0);

    // Prepare for the new generation. This should clear members of the current generation.
    species.preNewGeneration();

    EXPECT_EQ(species.getNumMembers(), 0);
    EXPECT_EQ(species.getBestGenome(), nullptr);
    EXPECT_EQ(species.getBestFitness(), 0);

    // Finalize the generation.
    species.postNewGeneration();

    EXPECT_EQ(species.getStagnantGenerationCount(), 1);

    // Prepare for the new generation. 
    species.preNewGeneration();

    // Try to add the genome to the species.
    EXPECT_TRUE(species.tryAddGenome(genome1, 2.f, 5.f, calcDistParams));

    EXPECT_EQ(species.getNumMembers(), 1);

    // Finalize the generation.
    species.postNewGeneration();

    EXPECT_EQ(species.getBestGenome(), genome1);
    EXPECT_EQ(species.getBestFitness(), 2.f);
    EXPECT_EQ(species.getStagnantGenerationCount(), 0);
}

TEST(Species, CreateSpeciesWithExistingGenome)
{
    using namespace NEAT;
    using GenomePtr = std::shared_ptr<Genome>;

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;
    GenomePtr initGenome = std::make_shared<Genome>(cinfo);

    // Create a species
    Species species(initGenome, 1.0f);

    EXPECT_EQ(species.getNumMembers(), 1);
    EXPECT_EQ(species.getBestGenome(), initGenome);
    EXPECT_EQ(species.getBestFitness(), 1.f);
    EXPECT_EQ(species.getStagnantGenerationCount(), 0);

    // Create a genome to add the species
    GenomePtr genome1 = std::make_shared<Genome>(*initGenome.get());

    // Mutate the genome
    {
        DefaultMutation::MutationParams mutParams;
        mutParams.m_weightMutationRate = 0.0f;
        mutParams.m_addEdgeMutationRate = 0.0f;
        mutParams.m_addNodeMutationRate = 1.0f;

        DefaultMutation mutator(mutParams);

        DefaultMutation::MutationOut mutOut;
        mutator.mutate(genome1.get(), mutOut);
    }

    // Set up calc distance params.
    Genome::CalcDistParams calcDistParams;
    {
        calcDistParams.m_disjointFactor = 1.0f;
        calcDistParams.m_weightFactor = 1.0f;
    }

    // We don't call preNewGeneration() here to keep the initGenom in the species' member.

    // Try to add the genome to the species
    EXPECT_TRUE(species.tryAddGenome(genome1, 0.5f, 5.f, calcDistParams));

    EXPECT_EQ(species.getNumMembers(), 2);

    // Finalize the generation.
    species.postNewGeneration();

    EXPECT_EQ(species.getStagnantGenerationCount(), 0);
    EXPECT_EQ(species.getBestGenome(), initGenome);
    EXPECT_EQ(species.getBestFitness(), 1.f);
}
