/*
* SpeciesChampionSelector.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <EvoAlgo/EvoAlgo.h>
#include <EvoAlgo/GeneticAlgorithms/NEAT/Generators/SpeciesChampionSelector.h>

using namespace NEAT;

SpeciesChampionSelector::SpeciesChampionSelector(float minMembersInSpeciesToCopyChampion)
    : m_minMembersInSpeciesToCopyChampion(minMembersInSpeciesToCopyChampion)
{
}

void SpeciesChampionSelector::updateSpecies(const SpeciesList& species, float bestFitness)
{
    m_species = &species;
    m_bestFitness = bestFitness;
}

void SpeciesChampionSelector::generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelector* /*genomeSelector*/)
{
    assert(numTotalGenomes >= numRemaningGenomes);

    using GenomePtr = std::shared_ptr<Genome>;

    m_generatedGenomes.clear();

    if (!m_species || numRemaningGenomes <= 0)
    {
        return;
    }

    // If the number of species is greater than the number of remaining genomes, sort selected genomes by their fitness and
    // take genomes in order of their fitness.
    bool canSelectAllChampions = true;
    const int numSpecies = (int)m_species->size();
    std::vector<float> fitnesses;
    if (numSpecies > numRemaningGenomes)
    {
        canSelectAllChampions = false;
        m_generatedGenomes.reserve(numRemaningGenomes);
        fitnesses.reserve(numRemaningGenomes);
    }
    else
    {
        m_generatedGenomes.reserve(numSpecies);
    }

    // Select genomes which are copied to the next generation unchanged.
    for (const auto& itr : *m_species)
    {
        const SpeciesPtr& species = itr.second;
        if (!species->isReproducible())
        {
            continue;
        }

        Species::CGenomePtr best = species->getBestGenome();
        if (best)
        {
            const float fitness = species->getBestFitness();

            if (fitness >= m_bestFitness)
            {
                m_generatedGenomes.push_back(std::make_shared<Genome>(*best));
                continue;
            }

            if (species->getNumMembers() >= m_minMembersInSpeciesToCopyChampion)
            {
                // Copy the champion.
                GenomePtr copiedGenome = std::make_shared<Genome>(*best);

                if (canSelectAllChampions)
                {
                    m_generatedGenomes.push_back(copiedGenome);
                }
                else
                {
                    // We cannot select all the champions. Sort them by fitness and only select good ones.

                    if (fitnesses.size() == 0)
                    {
                        m_generatedGenomes.push_back(copiedGenome);
                        fitnesses.push_back(fitness);
                        continue;
                    }

                    // Find the place where this genome can go in the sorted order by fitness.
                    auto gItr = m_generatedGenomes.begin();
                    auto fItr = fitnesses.begin();
                    for (; fItr != fitnesses.end(); gItr++, fItr++)
                    {
                        if (fitness > *fItr)
                        {
                            fitnesses.insert(fItr, fitness);
                            m_generatedGenomes.insert(gItr, copiedGenome);

                            if ((int)m_generatedGenomes.size() > numRemaningGenomes)
                            {
                                // Remove the worst genome and its fitness.
                                fitnesses.pop_back();
                                m_generatedGenomes.pop_back();
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    assert((int)m_generatedGenomes.size() <= numRemaningGenomes);
}
