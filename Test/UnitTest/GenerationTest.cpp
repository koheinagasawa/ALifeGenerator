/*
* GenomeTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/Generation.h>

namespace
{
    using namespace NEAT;

    class MyFitnessCalculator : public FitnessCalculatorBase
    {
    public:
        virtual float calcFitness(const Genome& genome) const override
        {
            return 1.0;
        }
    };
}

TEST(Generation, CreateGeneration)
{
    using namespace NEAT;

    InnovationCounter innovCounter;
    MyFitnessCalculator calclator;
    Generation::Cinfo cinfo;
    cinfo.m_numGenomes = 100;
    cinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;
    cinfo.m_genomeCinfo.m_numInputNodes = 3;
    cinfo.m_genomeCinfo.m_numOutputNodes = 3;
    cinfo.m_fitnessCalculator = &calclator;
    Generation generation(cinfo);

    EXPECT_EQ(generation.getNumGenomes(), 100);
    const Generation::GenomeData& gd = generation.getGenomes()[0];
    EXPECT_TRUE(gd.getGenome());
    EXPECT_EQ(gd.getFitness(), 0.f);
    EXPECT_EQ(gd.getSpeciesIndex(), -1);
    EXPECT_TRUE(gd.canReproduce());
    EXPECT_EQ(generation.getSpecies().size(), 1);
    EXPECT_EQ(generation.getId().val(), 0);
}

TEST(Generation, IncrementGeneration)
{
    using namespace NEAT;

    InnovationCounter innovCounter;
    MyFitnessCalculator calclator;
    Generation::Cinfo cinfo;
    cinfo.m_numGenomes = 100;
    cinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;
    cinfo.m_genomeCinfo.m_numInputNodes = 3;
    cinfo.m_genomeCinfo.m_numOutputNodes = 3;
    cinfo.m_fitnessCalculator = &calclator;
    Generation generation(cinfo);

    Generation::CreateNewGenParams params;
    generation.createNewGeneration(params);

    EXPECT_EQ(generation.getNumGenomes(), 100);
    const Generation::GenomeData& gd = generation.getGenomes()[0];
    EXPECT_TRUE(gd.getGenome());
    EXPECT_EQ(gd.getFitness(), 1.f);
    EXPECT_NE(gd.getSpeciesIndex(), -1);
    EXPECT_EQ(generation.getId().val(), 1);
}
