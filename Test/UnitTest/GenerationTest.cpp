/*
* GenomeTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

//#include <NEAT/Generation.h>
//
//namespace
//{
//    using namespace NEAT;
//
//    class MyFitnessCalculator : public FitnessCalculatorBase
//    {
//    public:
//        virtual float calcFitness(const Genome& genome) const override
//        {
//            genome.evaluate({ 1.f, 1.f, 1.f });
//
//            const Genome::Network::NodeIds& outputNodes = genome.getNetwork()->getOutputNodes();
//            float fitness = 0.f;
//            for (NodeId node : outputNodes)
//            {
//                fitness += genome.getNetwork()->getNode(node).getValue();
//            }
//            return std::max(0.f, fitness);
//        }
//    };
//}
//
//TEST(Generation, CreateGeneration)
//{
//    using namespace NEAT;
//
//    InnovationCounter innovCounter;
//    MyFitnessCalculator calclator;
//    Generation::Cinfo cinfo;
//    cinfo.m_numGenomes = 100;
//    cinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;
//    cinfo.m_genomeCinfo.m_numInputNodes = 3;
//    cinfo.m_genomeCinfo.m_numOutputNodes = 3;
//    cinfo.m_fitnessCalculator = &calclator;
//    Generation generation(cinfo);
//
//    EXPECT_EQ(generation.getNumGenomes(), 100);
//    for (int i = 0; i < generation.getNumGenomes(); i++)
//    {
//        const Generation::GenomeData& gd = generation.getGenomes()[i];
//        EXPECT_TRUE(gd.getGenome());
//        EXPECT_EQ(gd.getSpeciesId(), -1);
//        EXPECT_TRUE(gd.canReproduce());
//    }
//    EXPECT_EQ(generation.getAllSpecies().size(), 1);
//    EXPECT_TRUE(generation.getSpecies(SpeciesId(0)));
//    EXPECT_FALSE(generation.getSpecies(SpeciesId(0))->getBestGenome());
//    EXPECT_EQ(generation.getSpecies(SpeciesId(0))->getStagnantGenerationCount(), 0);
//    EXPECT_EQ(generation.getSpecies(SpeciesId(0))->getNumMembers(), 0);
//    EXPECT_EQ(&generation.getFitnessCalculator(), &calclator);
//    EXPECT_EQ(generation.getId().val(), 0);
//}
//
//TEST(Generation, IncrementGeneration)
//{
//    using namespace NEAT;
//
//    InnovationCounter innovCounter;
//    MyFitnessCalculator calclator;
//    Genome::Activation activation = [](float value) { return value; };
//
//    Generation::Cinfo cinfo;
//    {
//        cinfo.m_numGenomes = 100;
//        cinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;
//        cinfo.m_genomeCinfo.m_numInputNodes = 3;
//        cinfo.m_genomeCinfo.m_numOutputNodes = 3;
//        cinfo.m_genomeCinfo.m_defaultActivation = &activation;
//        cinfo.m_maxWeight = 3.f;
//        cinfo.m_minWeight = -3.f;
//        cinfo.m_fitnessCalculator = &calclator;
//    }
//
//    Generation generation(cinfo);
//
//    Generation::CreateNewGenParams params;
//    generation.createNewGeneration(params);
//
//    EXPECT_EQ(generation.getNumGenomes(), 100);
//    for (int i = 0; i < generation.getNumGenomes(); i++)
//    {
//        const Generation::GenomeData& gd = generation.getGenomes()[i];
//        EXPECT_TRUE(gd.getGenome());
//        EXPECT_NE(gd.getSpeciesId(), -1);
//    }
//    EXPECT_EQ(generation.getId().val(), 1);
//
//    generation.createNewGeneration(params);
//
//    EXPECT_EQ(generation.getNumGenomes(), 100);
//    EXPECT_EQ(generation.getId().val(), 2);
//}
