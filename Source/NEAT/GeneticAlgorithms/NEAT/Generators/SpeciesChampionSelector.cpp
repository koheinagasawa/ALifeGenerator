/*
* SpeciesChampionSelector.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Generators/SpeciesChampionSelector.h>

using namespace NEAT;

SpeciesChampionSelector::SpeciesChampionSelector(float minMembersInSpeciesToCopyChampion)
    : m_minMembersInSpeciesToCopyChampion(minMembersInSpeciesToCopyChampion)
{
}

void SpeciesChampionSelector::generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelector* /*genomeSelector*/)
{
    using GenomePtr = std::shared_ptr<Genome>;

    // Select genomes which are copied to the next generation unchanged.
    for (const auto& itr : *m_species)
    {
        const SpeciesPtr& species = itr.second;
        if (!species->isReproducible())
        {
            continue;
        }

        if (species->getNumMembers() >= m_minMembersInSpeciesToCopyChampion)
        {
            Species::CGenomePtr best = species->getBestGenome();
            if (best)
            {
                // Copy the champion.
                GenomePtr copiedGenome = std::make_shared<Genome>(*best);
                m_generatedGenomes.push_back(copiedGenome);
            }
        }
    }
}
