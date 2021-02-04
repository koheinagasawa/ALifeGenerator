/*
* Species.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/Species.h>

using namespace NEAT;

Species::Species(const Genome& initialRepresentative)
    : m_representative(initialRepresentative)
{
}

void Species::preNewGeneration(PseudoRandom* randomIn)
{
    assert(hasMember());

    // Select a new representative
    {
        RandomGenerator* random = randomIn ? randomIn : &PseudoRandom::getInstance();
        int index = random->randomInteger(0, m_members.size() - 1);
        GenomePtr representative = m_members[index];
        m_representative = *representative.get();
    }

    m_members.clear();
}

bool Species::tryAddGenome(GenomePtr genome, float distanceThreshold, const Genome::CalcDistParams& params)
{
    // Calculate distance between the representative.
    const float distance = Genome::calcDistance(*genome.get(), m_representative, params);

    if (distance <= distanceThreshold)
    {
        m_members.push_back(genome);
        return true;
    }

    return false;
}
