/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/Generation.h>

using namespace NEAT;

Generation::Generation(const Cinfo& cinfo)
    : m_id(GenerationId(0))
{
    assert(cinfo.m_numGenomes > 0);
    assert(cinfo.m_minWeight <= cinfo.m_maxWeight);

    PseudoRandom& random = cinfo.m_random ? *cinfo.m_random : PseudoRandom::getInstance();

    // Create genomes of the first generation.
    m_genomes = std::make_shared<Genomes>();
    m_genomes->reserve(cinfo.m_numGenomes);
    for (int i = 0; i < cinfo.m_numGenomes; i++)
    {
        GenomeData gd;
        gd.m_genome = std::make_shared<Genome>(cinfo.m_genomeCinfo);
        gd.m_id = GenomeId(i);

        // Randomize edge weights
        const Genome::Network* network = gd.m_genome->getNetwork();
        for (auto itr : network->getEdges())
        {
            gd.m_genome->setEdgeWeight(itr.first, random.randomReal(cinfo.m_minWeight, cinfo.m_maxWeight));
        }

        m_genomes->push_back(gd);
    }
}
