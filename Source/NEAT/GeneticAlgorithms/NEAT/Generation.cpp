/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Generation.h>
#include <NEAT/GeneticAlgorithms/NEAT/Selectors/SpeciesBasedGenomeSelector.h>
#include <NEAT/GeneticAlgorithms/NEAT/Generators/SpeciesChampionSelector.h>
#include <NEAT/GeneticAlgorithms/Base/Selectors/UniformGenomeSelector.h>
#include <NEAT/GeneticAlgorithms/Base/Generators/GenomeCloner.h>

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
        SpeciesId newSpeciesId = m_speciesIdGenerator.getNewId();
        SpeciesPtr newSpecies = std::make_shared<Species>(*std::static_pointer_cast<const Genome>(representative.getGenome()));
        m_species.insert({ newSpeciesId, newSpecies });

        // Assign this species to all the genomes
        m_genomesSpecies.reserve(m_genomes->size());
        for (const auto& genome : *m_genomes)
        {
            m_genomesSpecies.insert({ genome.getId(), newSpeciesId });
            newSpecies->addGenome(std::static_pointer_cast<const Genome>(genome.getGenome()), 0.f);
        }
    }

    // Create generators.
    {
        m_generators.reserve(3);

        // Create champion selector.
        m_speciesChampSelector = std::make_shared<SpeciesChampionSelector>(cinfo.m_minMembersInSpeciesToCopyChampion);
        m_generators.push_back(m_speciesChampSelector);

        // Create cross-over delegate.
        m_generators.push_back(std::make_unique<DefaultCrossOver>(cinfo.m_crossOverParams));

        // Create genome cloner.
        m_generators.push_back(std::make_unique<GenomeCloner<Genome>>());
    }

    // Create modifiers.
    {
        // Create mutator.
        m_mutator = std::make_shared<DefaultMutation>(cinfo.m_mutationParams);
        m_modifiers.push_back(m_mutator);
    }

    // Calculate initial fitness of genomes.
    calcFitness();
}

auto Generation::getGenomesInFitnessOrder() const->GenomeDatas
{
    // Copy genomes.
    GenomeDatas genomesOut = *m_genomes;

    // Sort copied genomes.
    std::sort(genomesOut.begin(), genomesOut.end(), [](const GenomeData& a, const GenomeData& b)
        {
            return a.getFitness() > b.getFitness();
        });

    return genomesOut;
}

auto Generation::getAllSpeciesInBestFitnessOrder() const->std::vector<SpeciesPtr>
{
    std::vector<SpeciesPtr> speciesOut;
    speciesOut.reserve(m_species.size());

    // Copy species.
    for (auto itr : m_species)
    {
        speciesOut.push_back(itr.second);
    }

    // Sort copied species.
    std::sort(speciesOut.begin(), speciesOut.end(), [](const SpeciesPtr& s1, const SpeciesPtr& s2)
        {
            return s1->getBestFitness() > s2->getBestFitness();
        });

    return speciesOut;
}

void Generation::preUpdateGeneration()
{
    // Update species in the champion selector.
    m_speciesChampSelector->updateSpecies(getAllSpecies());

    // Clear mutator.
    m_mutator->reset();

    // Clear protection of all genomes
    for (GenomeData& gd : *m_genomes)
    {
        gd.setProtected(false);
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
        s->preNewGeneration();
    }

    using CGenomePtr = std::shared_ptr<const Genome>;

    // Assign each genome to a species.
    for (const GenomeData& gd : *m_genomes)
    {
        // Try to find a species.
        const CGenomePtr genome = std::static_pointer_cast<const Genome>(gd.getGenome());

        auto itr = m_species.begin();
        for (; itr != m_species.end(); itr++)
        {
            SpeciesPtr& s = itr->second;
            if (s->tryAddGenome(genome, gd.getFitness(), m_params.m_speciationDistanceThreshold, m_params.m_calcDistParams))
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
            SpeciesPtr newSpecies = std::make_shared<Species>(genome, gd.getFitness());
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
        s->postNewGeneration(m_randomGenerator);

        // Mark stagnant species non-reproducible.
        // We don't do it if there is only one species because genome selection in the next generation
        // relied on that there is at least one reproducible species.
        if (m_species.size() > 1)
        {
            bool reproducible = s->getStagnantGenerationCount() < m_params.m_maxStagnantCount;
            s->setReproducible(reproducible);
        }
    }

    // Sort genomes by species id.
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
    std::shared_ptr<SpeciesBasedGenomeSelector> selector = std::make_shared<SpeciesBasedGenomeSelector>(*m_genomes, m_species, m_genomesSpecies);
    if(selector->getNumGenomes() > 0)
    {
        selector->setInterSpeciesSelectionRate(m_params.m_interSpeciesCrossOverRate);
        return std::static_pointer_cast<GenomeSelector>(selector);
    }

    // DefaultGenomeSelector failed to set up. This must mean all genomes have zero fitness.
    // Create uniform selector instead.
    WARN("All genomes have zero fitness. Use a uniform selector.");
    return std::static_pointer_cast<GenomeSelector>(std::make_shared<UniformGenomeSelector>(getGenomeData()));
}

bool Generation::isSpeciesReproducible(SpeciesId speciesId) const
{
    return m_species.at(speciesId)->isReproducible();
}

