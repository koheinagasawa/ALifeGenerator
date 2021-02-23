/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/Generation.h>
#include <NEAT/DefaultGenomeSelector.h>
#include <NEAT/SpeciesChampionSelector.h>
#include "UniformGenomeSelector.h"

using namespace NEAT;

Generation::Generation(const Cinfo& cinfo)
    : GenerationBase(GenerationId(0), cinfo.m_numGenomes, cinfo.m_fitnessCalculator, cinfo.m_random ? cinfo.m_random : &PseudoRandom::getInstance())
    , m_params(cinfo.m_generationParams)
{
    // Allocate a buffer for genomes of the first generation.
    m_genomes = std::make_shared<GenomeDatas>();
    m_genomes->reserve(cinfo.m_numGenomes);

    assert(cinfo.m_minWeight <= cinfo.m_maxWeight);

    // Create one genome which is used as an archetype for other genomes.
    const Genome archetypeGenome(cinfo.m_genomeCinfo);

    for (int i = 0; i < cinfo.m_numGenomes; i++)
    {
        // Create a genome.
        GenomePtr genome = std::make_shared<Genome>(archetypeGenome);

        // Randomize edge weights.
        const Genome::Network* network = genome->getNetwork();
        for (auto itr : network->getEdges())
        {
            genome->setEdgeWeight(itr.first, m_randomGenerator->randomReal(cinfo.m_minWeight, cinfo.m_maxWeight));
        }

        // Add the genome.
        m_genomes->push_back(GenomeData(genome, GenomeId(i)));
    }

    init(cinfo);
}

Generation::Generation(const Genomes& genomes, const Cinfo& cinfo)
    : GenerationBase(GenerationId(0), (int)genomes.size(), cinfo.m_fitnessCalculator, cinfo.m_random ? cinfo.m_random : &PseudoRandom::getInstance())
    , m_params(cinfo.m_generationParams)
{
    assert((int)genomes.size() == m_numGenomes);

    // Allocate a buffer for genomes of the first generation.
    m_genomes = std::make_shared<GenomeDatas>();
    m_genomes->reserve(genomes.size());

    // Create GenomeData for each given genome.
    GenomeId id(0);
    for (const GenomePtr& genome : genomes)
    {
        m_genomes->push_back(GenomeData(genome, id));
        id = id.val() + 1;
    }

    init(cinfo);
}
void Generation::init(const Cinfo& cinfo)
{
    // Create one species.
    {
        // Select a genome randomly and use it as representative of the species.
        const GenomeData& representative = (*m_genomes)[m_randomGenerator->randomInteger(0, m_genomes->size() - 1)];
        SpeciesId newSpecies = m_speciesIdGenerator.getNewId();
        m_species.insert({ newSpecies, std::make_shared<Species>(*static_cast<const Genome*>(representative.getGenome())) });
    }

    m_generators.reserve(3);

    // Create champion selector.
    m_generators.push_back(std::make_shared<SpeciesChampionSelector>(this, cinfo.m_minMembersInSpeciesToCopyChampion));

    // Create mutate delegate.
    m_generators.push_back(std::make_shared<DefaultMutation>(cinfo.m_mutationParams));

    // Create cross-over delegate.
    m_generators.push_back(std::make_unique<DefaultCrossOver>(cinfo.m_crossOverParams));

    // Calculate initial fitness of genomes.
    calcFitness();
}

void Generation::postUpdateGeneration()
{
    // Speciation

    // Remove stagnant species first.
    {
        auto itr = m_species.begin();
        while (itr != m_species.end())
        {
            const SpeciesPtr& s = itr->second;
            if (s->getStagnantGenerationCount() >= m_params.m_maxStagnantCount)
            {
                itr = m_species.erase(itr);
            }
            else
            {
                itr++;
            }
        }
    }

    // Prepare for the new generation of species.
    m_genomesSpecies.clear();
    for (auto& itr : m_species)
    {
        SpeciesPtr& s = itr.second;
        s->preNewGeneration(m_randomGenerator);
    }

    // Assign each genome to a species.
    for (GenomeData& gd : *m_genomes)
    {
        // Try to find a species.
        const Genome* genome = static_cast<const Genome*>(gd.getGenome());

        auto itr = m_species.begin();
        for (; itr != m_species.end(); itr++)
        {
            SpeciesPtr& s = itr->second;
            if (s->tryAddGenome(Species::CGenomePtr(genome), gd.getFitness(), m_params.m_speciationDistanceThreshold, m_params.m_calcDistParams))
            {
                m_genomesSpecies.insert({ gd.getId(), itr->first });
                break;
            }
        }

        if (itr == m_species.end())
        {
            // No species found. Create a new one for this genome.
            SpeciesId newSpeciesId = m_speciesIdGenerator.getNewId();
            m_genomesSpecies.insert({ gd.getId(), newSpeciesId });
            SpeciesPtr newSpecies = std::make_shared<Species>(Species::CGenomePtr(genome), gd.getFitness());
            m_species.insert({ newSpeciesId, newSpecies });
        }
    }

    // Remove empty species.
    {
        auto itr = m_species.begin();
        while (itr != m_species.end())
        {
            const SpeciesPtr& s = itr->second;
            if (s->getNumMembers() == 0)
            {
                itr = m_species.erase(itr);
            }
            else
            {
                itr++;
            }
        }
    }

    // Finalize new generation of species.
    for (auto& itr : m_species)
    {
        SpeciesPtr& s = itr.second;
        s->postNewGeneration();
    }

    // Sort genomes by species id
    std::sort(m_genomes->begin(), m_genomes->end(), [this](const GenomeData& g1, const GenomeData& g2)
        {
            SpeciesId s1 = getSpecies(g1.getId());
            SpeciesId s2 = getSpecies(g2.getId());
            return s1 != s2 ? s1 < s2 : g1.getFitness() > g2.getFitness();
        });
}

auto Generation::createSelector()->GenomeSelectorPtr
{
    // Create a DefaultGenomeSelector.
    std::shared_ptr<DefaultGenomeSelector> selector = std::make_unique<DefaultGenomeSelector>(this, *m_randomGenerator);

    // Try to set genomes.
    bool res = selector->setGenomes(*m_prevGenGenomes);

    if (!res)
    {
        // Failed to create GenomeSelector. This might mean that no genome is reproducible.
        // Mark all genomes reproducible and try again.
        selector->skipStagnantSpecies(false);
        res = selector->setGenomes(*m_prevGenGenomes);
    }

    if (res)
    {
        return GenomeSelectorPtr(selector.get());
    }

    // It failed again, this must mean all genomes have zero fitness.
    // Create uniform selector instead.
    WARN("All genomes have zero fitness. Use a uniform selector.");
    return std::make_unique<UniformGenomeSelector>(*m_randomGenerator);
}

bool Generation::isSpeciesReproducible(SpeciesId speciesId) const
{
    return m_species.at(speciesId)->getStagnantGenerationCount() < m_params.m_maxStagnantCount;
}

