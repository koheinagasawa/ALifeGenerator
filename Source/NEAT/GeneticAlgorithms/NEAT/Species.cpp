/*
* Species.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Species.h>

using namespace NEAT;

Species::Species(const Genome& initialRepresentative)
    : m_representative(initialRepresentative)
{
}

Species::Species(CGenomePtr initialMember, float fitness)
    : m_representative(*initialMember.get())
    , m_bestGenome(initialMember)
    , m_bestFitness(fitness)
{
    m_members.push_back(initialMember);
}

void Species::preNewGeneration()
{
    m_members.clear();
    m_bestFitness = 0.f;
    m_bestGenome = nullptr;
}

void Species::postNewGeneration(RandomGenerator* randomIn)
{
    if (m_bestFitness <= m_previousBestFitness)
    {
        // No improvement. Increment stagnant count.
        m_stagnantCount++;
    }
    else
    {
        // There is improvement. Reset stagnant count.
        m_previousBestFitness = m_bestFitness;
        m_stagnantCount = 0;
    }

    // Select a new representative
    if (getNumMembers() > 0)
    {
        RandomGenerator* random = randomIn ? randomIn : &PseudoRandom::getInstance();
        int index = random->randomInteger(0, m_members.size() - 1);
        CGenomePtr representative = m_members[index];
        m_representative = *representative.get();
    }
}

bool Species::tryAddGenome(CGenomePtr genome, float fitness, float distanceThreshold, const Genome::CalcDistParams& params)
{
    // Calculate distance between the representative.
    const float distance = Genome::calcDistance(*genome.get(), m_representative, params);

    if (distance <= distanceThreshold)
    {
        addGenome(genome, fitness);
        return true;
    }

    return false;
}

void Species::addGenome(CGenomePtr genome, float fitness)
{
    m_members.push_back(genome);

    // Update best fitness and genome.
    if (fitness > m_bestFitness)
    {
        m_bestFitness = fitness;
        m_bestGenome = genome;
    }
}
