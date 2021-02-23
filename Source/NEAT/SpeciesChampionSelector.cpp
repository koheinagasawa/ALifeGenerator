/*
* SpeciesChampionSelector.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/SpeciesChampionSelector.h>

using namespace NEAT;

SpeciesChampionSelector::SpeciesChampionSelector(const Generation* g, float minMembersInSpeciesToCopyChampion)
    : m_generation(g)
    , m_minMembersInSpeciesToCopyChampion(minMembersInSpeciesToCopyChampion)
{
}

void SpeciesChampionSelector::generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelector* genomeSelector)
{
    using SpeciesPtr = std::shared_ptr<Species>;
    using GenomePtr = std::shared_ptr<Genome>;

    // Select genomes which are copied to the next generation unchanged.
    for (auto& itr : m_generation->getAllSpecies())
    {
        const SpeciesPtr& species = itr.second;
        if (!m_generation->isSpeciesReproducible(itr.first))
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
