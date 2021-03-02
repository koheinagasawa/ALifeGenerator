/*
* GenomeTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/GeneticAlgorithms/NEAT/Generation.h>

namespace
{
    using namespace NEAT;

    // Custom fitness calculator.
    class MyFitnessCalculator : public FitnessCalculatorBase
    {
    public:
        virtual float calcFitness(const GenomeBase& genome) const override
        {
            genome.evaluate({ 1.f, 1.f, 1.f });

            const Genome::Network::NodeIds& outputNodes = genome.getNetwork()->getOutputNodes();
            float fitness = 0.f;
            for (NodeId node : outputNodes)
            {
                fitness += genome.getNetwork()->getNode(node).getValue();
            }
            return std::max(0.f, fitness);
        }
    };
}

TEST(Generation, CreateGeneration)
{
    using namespace NEAT;

    // Create a generation with 100 population.
    InnovationCounter innovCounter;
    Generation::Cinfo cinfo;
    cinfo.m_numGenomes = 100;
    cinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;
    cinfo.m_genomeCinfo.m_numInputNodes = 3;
    cinfo.m_genomeCinfo.m_numOutputNodes = 3;
    cinfo.m_fitnessCalculator = std::make_shared<MyFitnessCalculator>();
    Generation generation(cinfo);

    EXPECT_EQ(generation.getNumGenomes(), 100);
    for (int i = 0; i < generation.getNumGenomes(); i++)
    {
        const Generation::GenomeData& gd = generation.getGenomes()[i];
        EXPECT_TRUE(gd.getGenome());
        EXPECT_EQ(generation.getSpecies(gd.getId()), SpeciesId::invalid());
    }
    EXPECT_EQ(generation.getAllSpecies().size(), 1);
    EXPECT_TRUE(generation.getSpecies(SpeciesId(0)));
    EXPECT_FALSE(generation.getSpecies(SpeciesId(0))->getBestGenome());
    EXPECT_EQ(generation.getSpecies(SpeciesId(0))->getStagnantGenerationCount(), 0);
    EXPECT_EQ(generation.getSpecies(SpeciesId(0))->getNumMembers(), 0);
    EXPECT_TRUE(generation.isSpeciesReproducible(SpeciesId(0)));
    EXPECT_EQ(generation.getId().val(), 0);
}

TEST(Generation, IncrementGeneration)
{
    using namespace NEAT;

    // Create a generation with 20 population.
    InnovationCounter innovCounter;
    Genome::Activation activation = [](float value) { return value; };

    Generation::Cinfo cinfo;
    {
        cinfo.m_numGenomes = 20;
        cinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;
        cinfo.m_genomeCinfo.m_numInputNodes = 3;
        cinfo.m_genomeCinfo.m_numOutputNodes = 3;
        cinfo.m_genomeCinfo.m_defaultActivation = &activation;
        cinfo.m_maxWeight = 3.f;
        cinfo.m_minWeight = -3.f;
        cinfo.m_fitnessCalculator = std::make_shared<MyFitnessCalculator>();
    }

    Generation generation(cinfo);

    // Evolve the generation several times.
    generation.evolveGeneration();

    EXPECT_EQ(generation.getNumGenomes(), 20);

    // Make sure that all the genomes were assigned to a species.
    for (int i = 0; i < generation.getNumGenomes(); i++)
    {
        const Generation::GenomeData& gd = generation.getGenomes()[i];
        EXPECT_TRUE(gd.getGenome());
        EXPECT_NE(generation.getSpecies(gd.getId()), SpeciesId::invalid());
    }

    EXPECT_TRUE(generation.getAllSpecies().size() > 0);
    EXPECT_EQ(generation.getId().val(), 1);

    generation.evolveGeneration();

    EXPECT_EQ(generation.getNumGenomes(), 20);
    EXPECT_TRUE(generation.getAllSpecies().size() > 0);
    EXPECT_EQ(generation.getId().val(), 2);

    generation.evolveGeneration();

    EXPECT_EQ(generation.getNumGenomes(), 20);
    EXPECT_TRUE(generation.getAllSpecies().size() > 0);
    EXPECT_EQ(generation.getId().val(), 3);

    generation.evolveGeneration();

    EXPECT_EQ(generation.getNumGenomes(), 20);
    EXPECT_TRUE(generation.getAllSpecies().size() > 0);
    EXPECT_EQ(generation.getId().val(), 4);

    generation.evolveGeneration();

    EXPECT_EQ(generation.getNumGenomes(), 20);
    EXPECT_TRUE(generation.getAllSpecies().size() > 0);
    EXPECT_EQ(generation.getId().val(), 5);

    generation.evolveGeneration();

    EXPECT_EQ(generation.getNumGenomes(), 20);
    EXPECT_TRUE(generation.getAllSpecies().size() > 0);
    EXPECT_EQ(generation.getId().val(), 6);
}
