/*
* GenomeTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/Genome.h>

TEST(Genome, CreateGenome)
{
    NEAT::InnovationCounter innovCounter;
    NEAT::Genome::Cinfo cinfo;
    cinfo.m_numInputNodes = 2;
    cinfo.m_numOutputNodes = 2;
    cinfo.m_innovIdCounter = &innovCounter;

    NEAT::Genome genome(cinfo);

    const NEAT::Genome::Network* network = genome.getNetwork();

    EXPECT_TRUE(network);
    EXPECT_TRUE(network->validate());
    EXPECT_EQ(network->getNumNodes(), 4);
    EXPECT_EQ(network->getNumEdges(), 4);
    EXPECT_EQ(network->getOutputNodes().size(), 2);
}
