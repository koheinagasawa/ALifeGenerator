/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/Generation.h>
#include "DefaultGenomeSelector.h"

using namespace NEAT;

Generation::Generation(const Cinfo& cinfo)
    : GenerationBase(GenerationId(0), cinfo.m_numGenomes, cinfo.m_fitnessCalculator)
    , m_params(cinfo.m_generationParams)
{
    // Create genomes of the first generation.
    m_genomes = std::make_shared<GenomeDatas>();
    m_genomes->reserve(cinfo.m_numGenomes);

    assert(cinfo.m_minWeight <= cinfo.m_maxWeight);

    const Genome archetypeGenome(cinfo.m_genomeCinfo);
    m_randomGenerator = cinfo.m_random ? cinfo.m_random : &PseudoRandom::getInstance();

    for (int i = 0; i < cinfo.m_numGenomes; i++)
    {
        GenomePtr genome = std::make_shared<Genome>(archetypeGenome);

        // Randomize edge weights.
        const Genome::Network* network = genome->getNetwork();
        for (auto itr : network->getEdges())
        {
            genome->setEdgeWeight(itr.first, m_randomGenerator->randomReal(cinfo.m_minWeight, cinfo.m_maxWeight));
        }

        m_genomes->push_back(GenomeData(genome, GenomeId(i)));
    }

    init(cinfo);
}

Generation::Generation(const Genomes& genomes, const Cinfo& cinfo)
    : GenerationBase(GenerationId(0), (int)genomes.size(), cinfo.m_fitnessCalculator)
    , m_params(cinfo.m_generationParams)
{
    assert((int)genomes.size() == m_numGenomes);

    m_genomes = std::make_shared<GenomeDatas>();

    // Create GenomeData for each given genome.
    {
        m_genomes->reserve(genomes.size());
        GenomeId id(0);
        for (const GenomePtr& genome : genomes)
        {
            m_genomes->push_back(GenomeData(genome, id));
            id = id.val() + 1;
        }
    }

    init(cinfo);
}

void Generation::init(const Cinfo& cinfo)
{
    // Create one species.
    {
        const GenomeData& representative = (*m_genomes)[m_randomGenerator->randomInteger(0, m_genomes->size() - 1)];
        SpeciesId newSpecies = m_speciesIdGenerator.getNewId();
        m_species.insert({ newSpecies, std::make_shared<Species>(*static_cast<const Genome*>(representative.getGenome())) });
    }

    m_generators.push_back(std::make_shared<BestGenomeSelector>(this, cinfo.m_generationParams.m_minMembersInSpeciesToCopyChampion));

    // Create mutate delegate
    m_generators.push_back(std::make_shared<DefaultMutation>(cinfo.m_mutationParams));

    // Create cross over delegate
    m_generators.push_back(std::make_unique<DefaultCrossOver>(cinfo.m_crossOverParams));

    // Calculate initial fitness of genomes.
    calcFitness();
}

void BestGenomeSelector::generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelectorBase* genomeSelector)
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
                GenomePtr copiedGenome = std::make_shared<Genome>(*best);
                m_generatedGenomes.push_back(copiedGenome);
            }
        }
    }
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
            SpeciesId newSpecies = m_speciesIdGenerator.getNewId();
            m_genomesSpecies.insert({ gd.getId(), newSpecies });
            m_species.insert({ newSpecies, std::make_shared<Species>(*genome) });
            m_species[newSpecies]->tryAddGenome(Species::CGenomePtr(genome), gd.getFitness(), m_params.m_speciationDistanceThreshold, m_params.m_calcDistParams);
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
    std::shared_ptr<DefaultGenomeSelector> selector = std::make_unique<DefaultGenomeSelector>(this, *m_randomGenerator);
    {
        selector->setInterSpeciesCrossOverRate(m_params.m_interSpeciesCrossOverRate);

        bool res = selector->setGenomes(*m_prevGenGenomes);
        if (!res)
        {
            // Failed to create GenomeSelector. This means that no genome is reproducible.
            // Mark all genomes reproducible and try again.
            selector->skipStagnantSpecies(false);
            res = selector->setGenomes(*m_prevGenGenomes);

            assert(res);
        }
    }

    return GenomeSelectorPtr(selector.get());
}

void Generation::calcFitness()
{
    for (GenomeData& gd : *m_genomes)
    {
        gd.setFitness(m_fitnessCalculator->calcFitness(*gd.getGenome()));
    }
}

auto Generation::getSpecies(GenomeId genomeId) const->SpeciesId
{
    return m_genomesSpecies.at(genomeId);
}

bool Generation::isSpeciesReproducible(SpeciesId speciesId) const
{
    return m_species.at(speciesId)->getStagnantGenerationCount() < m_maxStagnantCount;
}

