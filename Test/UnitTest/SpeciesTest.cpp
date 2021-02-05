/*
* SpeciesTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/Species.h>

TEST(Species, AddGenomeToSpecies)
{
    using namespace NEAT;

    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;

    // Create a genome.
    Genome initGenome(cinfo);

    Species species(initGenome);

    EXPECT_FALSE(species.hasMember());

    using GenomePtr = std::shared_ptr<Genome>;

    GenomePtr genome1 = std::make_shared<Genome>(initGenome);

    Genome::MutationParams mutParams;
    mutParams.m_weightMutationRate = 0.0f;
    mutParams.m_addEdgeMutationRate = 0.0f;
    mutParams.m_addNodeMutationRate = 1.0f;

    Genome::MutationOut mutOut;
    genome1->mutate(mutParams, mutOut);

    Genome::CalcDistParams calcDistParams;
    calcDistParams.m_disjointFactor = 1.0f;
    calcDistParams.m_weightFactor = 1.0f;

    EXPECT_FALSE(species.tryAddGenome(genome1, 0.0001f, calcDistParams));
    EXPECT_FALSE(species.hasMember());
    EXPECT_TRUE(species.tryAddGenome(genome1, 5.f, calcDistParams));
    EXPECT_TRUE(species.hasMember());

    species.preNewGeneration();

    EXPECT_FALSE(species.hasMember());
}

