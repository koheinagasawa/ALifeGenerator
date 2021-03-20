/*
* SpeciesBasedGenomeSelector.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Selectors/SpeciesBasedGenomeSelector.h>

using namespace NEAT;

SpeciesBasedGenomeSelector::SpeciesData::SpeciesData(SpeciesPtr species)
    : m_species(species)
{
    m_cumulativeFitnesses.push_back(0.f);
}

SpeciesBasedGenomeSelector::SpeciesBasedGenomeSelector(const GenomeDatas& genomeData, const SpeciesList& species, const GenomeSpeciesMap& genomeSpeciesMap, PseudoRandom* random)
    : GenomeSelector()
    , m_random(random ? *random : PseudoRandom::getInstance())
{
    const int numGenomes = (int)genomeData.size();
    assert(numGenomes > 0);

    auto getSpeciesId = [&genomeSpeciesMap](const GenomeData& g)
    {
        return genomeSpeciesMap.find(g.getId()) != genomeSpeciesMap.end() ? genomeSpeciesMap.at(g.getId()) : SpeciesId::invalid();
    };

#ifdef _DEBUG
    // Make sure that genomes are sorted by species id.
    {
        SpeciesId curId = getSpeciesId(genomeData[0]);
        for (const GenomeData& g : genomeData)
        {
            SpeciesId id = getSpeciesId(g);
            if (curId != id)
            {
                assert(curId < id);
                curId = id;
            }
        }
    }
#endif

    m_speciesData.reserve(species.size());

    // Helper functions to calculate factor for fitness sharing.
    // Genome's fitness is going to be normalized by the number of members in its species.
    auto calcFitnessSharingFactor = [this](SpeciesPtr species)->float
    {
        return species ? 1.f / (float)species->getNumMembers() : 1.0f;
    };

    auto addGenomes = [&]()
    {
        SpeciesId currentSpeciesId = SpeciesId::invalid();
        SpeciesPtr currentSpecies = nullptr;
        SpeciesData* speciesData = nullptr;

        // Calculate adjusted fitness for each genome and sum up those fitnesses.
        for (const GenomeData& g : genomeData)
        {
            const SpeciesId sId = getSpeciesId(g);
            if (!sId.isValid() || !species.at(sId)->isReproducible() || g.getFitness() == 0)
            {
                continue;
            }

            assert(g.getFitness() > 0);

            if (currentSpeciesId != sId)
            {
                // This genome is in a new species.
                currentSpeciesId = sId;
                currentSpecies = species.at(currentSpeciesId);

                m_speciesData.push_back(SpeciesData(currentSpecies));
                speciesData = &m_speciesData.back();
            }
            else
            {
                m_hasSpeciesMoreThanOneMember = true;
            }

            assert(speciesData);

            speciesData->m_genomes.push_back(&g);

            m_numGenomes++;
        }
    };

    addGenomes();

    if (m_numGenomes == 0)
    {
        // There was no genomes which can be reproducible or has positive fitness.
        // Try again without skipping stagnant species.
        m_skipStagnantSpecies = false;
        addGenomes();

        if (m_numGenomes == 0)
        {
            WARN("Failed to setup DefaultGenomeSelector because all genomes have zero fitness.");
        }
    }

    // Remove the least fit genomes in each species from selection
    for (auto& sData : m_speciesData)
    {
        // Sort genomes by fitness
        std::sort(sData.m_genomes.begin(), sData.m_genomes.end(), [](const GenomeData*& g1, const GenomeData*& g2)
            {
                return g1->getFitness() > g2->getFitness();
            });

        // Remove the least fit genome unless the species has less than three members or
        // the least fit genome has the same fitness as a genome at mean
        if (sData.m_genomes.size() > 2 && sData.m_genomes.back()->getFitness() < sData.m_genomes[sData.m_genomes.size() / 2]->getFitness())
        {
            sData.m_genomes.pop_back();
        }

        float fitnessSharingFactor = 1.f / (float)sData.m_species->getNumMembers();
        for (int i = 0; i < (int)sData.m_genomes.size(); i++)
        {
            // Calculate cumulative sum of fitness of the species' members.
            float fitness = sData.m_genomes[i]->getFitness() * fitnessSharingFactor;
            sData.m_cumulativeFitnesses.push_back(sData.m_cumulativeFitnesses.back() + fitness);
            m_totalFitness += fitness;
        }
    }

}

void SpeciesBasedGenomeSelector::setSpeciesPopulations(int numGenomesToSelect)
{
    assert(numGenomesToSelect > 0);

    // Set population of all species zero once.
    for (auto& sData : m_speciesData)
    {
        sData.m_population = 0;
        sData.m_remainingPopulation = 0;
    }

    m_currentSpeciesDataIndex = 0;

    if (numGenomesToSelect <= 0 || 
        (!hasSpeciesMoreThanOneMember() && m_mode == GenomeSelector::SELECT_TWO_GENOMES) ||
        m_totalFitness == 0.f)
    {
        return;
    }

    int remainingGenomes = numGenomesToSelect;
    m_numInterSpeciesSelection = (m_mode == GenomeSelector::SELECT_ONE_GENOME) ? 0 : (int)(numGenomesToSelect * m_interSpeciesSelectionRate);

    // Select at least one genome by inter-species selection when m_interSpeciesSelectionRate is non-zero.
    if (m_mode == GenomeSelector::SELECT_TWO_GENOMES && m_numInterSpeciesSelection == 0 && m_interSpeciesSelectionRate > 0)
    {
        m_numInterSpeciesSelection = 1;
    }

    remainingGenomes -= m_numInterSpeciesSelection;

    // Distribute population to species based on the sum of fitness of its members.
    float fitnessScale = 1.0f;
    while (remainingGenomes > 0)
    {
        for (auto& sData : m_speciesData)
        {
            // Skip species which has only one genome when for selection by two genomes at once.
            if (m_mode == GenomeSelector::SELECT_TWO_GENOMES && sData.getNumGenomes() < 2)
            {
                continue;
            }

            int population = int(sData.getSumFitness() * fitnessScale / m_totalFitness);

            // Make sure that we don't exceed the total number of genomes.
            if (population > remainingGenomes)
            {
                population = remainingGenomes;
            }

            sData.m_population += population;
            remainingGenomes -= population;
            assert(remainingGenomes >= 0);

            if (remainingGenomes == 0)
            {
                break;
            }
        }

        fitnessScale *= 2.f;
    }

    // Set remaining population
    for (auto& sData : m_speciesData)
    {
        sData.m_remainingPopulation = sData.m_population;
    }

    // Set the current species data index to the first species which has population
    while (m_speciesData[m_currentSpeciesDataIndex].m_population == 0)
    {
        m_currentSpeciesDataIndex++;
        if (m_currentSpeciesDataIndex == (int)m_speciesData.size())
        {
            // No species has population. This means all the selections are going to be inter-species selection.
            break;
        }
    }

    // Calculate cumulative fitness of species when we will need inter species selection
    if (m_numInterSpeciesSelection)
    {
        m_cumulativeSpeciesFitness.clear();
        m_cumulativeSpeciesFitness.reserve(m_speciesData.size() + 1);
        m_cumulativeSpeciesFitness.push_back(0);
        for (const auto& sData : m_speciesData)
        {
            m_cumulativeSpeciesFitness.push_back(m_cumulativeSpeciesFitness.back() + sData.m_cumulativeFitnesses.back());
        }
    }
}

void SpeciesBasedGenomeSelector::decrementPopulationOfCurrentSpecies()
{
    SpeciesData& sData = m_speciesData[m_currentSpeciesDataIndex];
    sData.m_remainingPopulation -= 1;
    if (sData.m_remainingPopulation == 0)
    {
        do
        {
            m_currentSpeciesDataIndex++;
        } while (m_currentSpeciesDataIndex < (int)m_speciesData.size() && m_speciesData[m_currentSpeciesDataIndex].m_population == 0);
    }
}

void SpeciesBasedGenomeSelector::preSelection(int numGenomesToSelect, SelectionMode mode)
{
    m_mode = mode;
    setSpeciesPopulations(numGenomesToSelect);
}

void SpeciesBasedGenomeSelector::postSelection()
{
    if (m_numGenomes == 0)
    {
        return;
    }

#ifdef _DEBUG
    {
        for (const auto& sData : m_speciesData)
        {
            assert(sData.m_remainingPopulation == 0);
        }
    }
#endif

}

auto SpeciesBasedGenomeSelector::selectGenomeImpl()->const GenomeData*
{
    if (m_numGenomes == 0)
    {
        return nullptr;
    }

    assert(m_currentSpeciesDataIndex >= 0 && m_currentSpeciesDataIndex < (int)m_speciesData.size());
    assert(m_speciesData.size() > 0);

    const SpeciesData& sData = m_speciesData[m_currentSpeciesDataIndex];

    const std::vector<float>& fitnesses = sData.m_cumulativeFitnesses;
    assert(sData.getNumGenomes() + 1 == fitnesses.size());
    assert(sData.m_remainingPopulation > 0);

    // std::uniform_real_distribution should return [min, max), but if we call randomReal(m_sumFitness[start], m_sumFitness[end])
    // we see v == m_sumFitness[end] here for some reason. That's why we have to calculate nexttoward of max here to avoid unintentional
    // calculation later.
    const float v = m_random.randomReal(fitnesses[0], std::nexttoward(fitnesses.back(), -1.f));
    for (int i = 0; i < (int)fitnesses.size(); i++)
    {
        if (v < fitnesses[i + 1])
        {
            return sData.m_genomes[i];
        }
    }

    assert(0);
    return nullptr;
}

auto SpeciesBasedGenomeSelector::selectGenome()->const GenomeData*
{
    assert(m_mode == GenomeSelector::SELECT_ONE_GENOME);

    const GenomeData* genomeOut =  selectGenomeImpl();
    if (genomeOut)
    {
        decrementPopulationOfCurrentSpecies();
    }

    return genomeOut;
}

void SpeciesBasedGenomeSelector::selectTwoGenomes(const GenomeData*& g1, const GenomeData*& g2)
{
    assert(m_mode == GenomeSelector::SELECT_TWO_GENOMES);

    g1 = nullptr;
    g2 = nullptr;

    if (m_numGenomes < 2)
    {
        return;
    }

    assert(hasSpeciesMoreThanOneMember());
    assert(m_speciesData.size() > 0);
    assert(m_currentSpeciesDataIndex >= 0);

    if (m_currentSpeciesDataIndex < (int)m_speciesData.size())
    {
        // Intra-species selection

        // Skip genomes with less than 2 members.
        while (m_speciesData[m_currentSpeciesDataIndex].getNumGenomes() < 2)
        {
            m_currentSpeciesDataIndex++;
        }

        SpeciesData& sData = m_speciesData[m_currentSpeciesDataIndex];

        if (sData.getNumGenomes() == 2)
        {
            // There are only two genomes in this species.
            g1 = sData.m_genomes[0];
            g2 = sData.m_genomes[1];
        }
        else
        {
            // Select g1 and g2 among the species.
            g1 = selectGenomeImpl();
            g2 = g1;
            while (g1 == g2)
            {
                g2 = selectGenomeImpl();
            }
        }

        decrementPopulationOfCurrentSpecies();
    }
    else
    {
        // Inter species selection.

        assert(m_cumulativeSpeciesFitness.size() == m_speciesData.size() + 1);

        auto select = [this]()->const GenomeData*
        {
            // std::uniform_real_distribution should return [min, max), but if we call randomReal(m_sumFitness[start], m_sumFitness[end])
            // we see v == m_sumFitness[end] here for some reason. That's why we have to calculate nexttoward of max here to avoid unintentional
            // calculation later.
            float v = m_random.randomReal(m_cumulativeSpeciesFitness[0], std::nexttoward(m_cumulativeSpeciesFitness.back(), -1.f));
            for (int i = 0; i < (int)m_speciesData.size(); i++)
            {
                if (v < m_cumulativeSpeciesFitness[i + 1])
                {
                    if (i > 0)
                    {
                        v -= m_cumulativeSpeciesFitness[i];
                    }

                    const SpeciesData& sData = m_speciesData[i];

                    for (int j = 0; j < (int)sData.m_genomes.size(); j++)
                    {
                        if (v < sData.m_cumulativeFitnesses[j + 1])
                        {
                            return sData.m_genomes[j];
                        }
                    }
                }
            }
            assert(0);
            return nullptr;
        };

        g1 = select();
        g2 = g1;
        while (g2 == g1)
        {
            g2 = select();
        }
    }
}
