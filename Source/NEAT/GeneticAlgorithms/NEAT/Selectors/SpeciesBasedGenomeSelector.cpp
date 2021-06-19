/*
* SpeciesBasedGenomeSelector.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Selectors/SpeciesBasedGenomeSelector.h>

using namespace NEAT;

//
// SpeciesBasedGenomeSelector::SpeciesData
//

SpeciesBasedGenomeSelector::SpeciesData::SpeciesData(SpeciesPtr species)
    : m_species(species)
{
    m_cumulativeFitnesses.push_back(0.f);
}

//
// SpeciesBasedGenomeSelector
//

SpeciesBasedGenomeSelector::SpeciesBasedGenomeSelector(const GenomeDatas& genomeData, const SpeciesList& species, const GenomeSpeciesMap& genomeSpeciesMap, PseudoRandom* random)
    : GenomeSelector()
    , m_random(random ? *random : PseudoRandom::getInstance())
{
    const int numGenomes = (int)genomeData.size();
    assert(numGenomes > 0);

    auto getSpeciesId = [&genomeSpeciesMap](const GenomeData& g)
    {
        return (genomeSpeciesMap.find(g.getId()) != genomeSpeciesMap.end()) ? genomeSpeciesMap.at(g.getId()) : SpeciesId::invalid();
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

    // Collect species which is reproducible.
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
                // Skip species who is marked as not reproducible and skip genome whose fitness is zero.
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
    }

    if (m_numGenomes == 0)
    {
        // There was no genomes which can be reproducible or has positive fitness.
        WARN("Failed to setup DefaultGenomeSelector because all genomes have zero fitness.");
    }

    //// Sort species data in the best fitness order
    //std::sort(m_speciesData.begin(), m_speciesData.end(), [](const SpeciesData& s1, const SpeciesData& s2)
    //    {
    //        return s1.m_species->getBestFitness() > s2.m_species->getBestFitness();
    //    });

    // Remove the least fit genomes in each species from selection
    for (auto& sData : m_speciesData)
    {
        // Sort genomes by fitness
        std::sort(sData.m_genomes.begin(), sData.m_genomes.end(), [](const GenomeData*& g1, const GenomeData*& g2)
            {
                return g1->getFitness() > g2->getFitness();
            });

        const float fitnessSharingFactor = 1.f / (float)sData.m_species->getNumMembers();

        // Remove the least fit genome(s) unless the species has less than three members or
        // the least fit genome(s) has the same fitness as a genome at the middle.
        {
            const float leastFitness = sData.m_genomes.back()->getFitness();
            if (sData.m_genomes.size() > 2 && leastFitness < sData.m_genomes[sData.m_genomes.size() / 2]->getFitness())
            {
                do 
                {
                    sData.m_genomes.pop_back();
                } while (sData.m_genomes.back()->getFitness() == leastFitness);
            }
        }

        for (int i = 0; i < (int)sData.m_genomes.size(); i++)
        {
            // Calculate cumulative sum of fitness of the species' members.
            float fitness = sData.m_genomes[i]->getFitness() * fitnessSharingFactor;
            sData.m_cumulativeFitnesses.push_back(sData.m_cumulativeFitnesses.back() + fitness);
            m_totalFitness += fitness;
        }
    }

}

void SpeciesBasedGenomeSelector::distributeSpeciesPopulations(int numGenomesToSelect)
{
    assert(numGenomesToSelect > 0);

    // Set population of all species zero once.
    for (auto& sData : m_speciesData)
    {
        sData.m_population = 0;
        sData.m_remainingPopulation = 0;
    }

    // Reset the current species index.
    m_currentSpeciesDataIndex = 0;

    if (numGenomesToSelect <= 0 || m_totalFitness == 0.f)
    {
        // Nothing to select. Abort.
        return;
    }

    int remainingGenomes = numGenomesToSelect;

    // Set the number of genomes to select by inter species selection.
    if (m_mode == GenomeSelector::SELECT_ONE_GENOME || m_speciesData.size() == 1)
    {
        m_numInterSpeciesSelection = 0;
    }
    else
    {
        if (hasSpeciesMoreThanOneMember())
        {
            m_numInterSpeciesSelection = (int)numGenomesToSelect * m_interSpeciesSelectionRate;
        }
        else
        {
            // We don't have any species which we can select two species from.
            // Then just select all species as inter species selection.
            m_numInterSpeciesSelection = numGenomesToSelect;
        }

        // Select at least one genome by inter-species selection when m_interSpeciesSelectionRate is non-zero.
        if (m_numInterSpeciesSelection == 0 && m_interSpeciesSelectionRate > 0)
        {
            m_numInterSpeciesSelection = 1;
        }
    }

    remainingGenomes -= m_numInterSpeciesSelection;

    // Utility function to tell if a species is applicable for the current selection.
    auto speciesNotApplicable = [&](const SpeciesData& sData)->bool
    {
        // Skip species which has only one genome when for selection by two genomes at once.
        return m_mode == GenomeSelector::SELECT_TWO_GENOMES && sData.getNumGenomes() < 2;
    };

    // Calculate the total fitness of species which are applicable for this selection.
    float totalFitness = 0;
    for (int i = 0; i < (int)m_speciesData.size(); i++)
    {
        const SpeciesData& sData = m_speciesData[i];

        if(speciesNotApplicable(sData))
        {
            continue;
        }

        totalFitness += sData.m_cumulativeFitnesses.back();
    }

    // Distribute population to species based on the sum of fitness of its members.
    {
        int assignedGenomes = 0;
        std::vector<std::pair<int, float>> residues; // Intermediate data to store species index and decimal part of the distributed population.
        residues.reserve(m_speciesData.size());
        for (int i = 0; i < (int)m_speciesData.size(); i++)
        {
            SpeciesData& sData = m_speciesData[i];

            if (speciesNotApplicable(sData))
            {
                residues.push_back({ i, 0.f });
                continue;
            }

            const float populationF = sData.getSumFitness() / totalFitness * remainingGenomes; // Float value of population.
            int population = int(populationF); // Integer value of population.
            residues.push_back({ i, populationF - (float)population }); // Remember the decimal part.

            sData.m_population = population;
            assignedGenomes += population;
            assert(assignedGenomes <= remainingGenomes);
        }

        assert(assignedGenomes <= remainingGenomes && (remainingGenomes - assignedGenomes) < (int)m_speciesData.size());

        if (assignedGenomes < remainingGenomes)
        {
            // There are still remaining population. Distribute it based on the decimal part of each species.

            // Sort species by the decimal part. Species with larger decimals have an extra population.
            std::sort(residues.begin(), residues.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b)
                {
                    return a.second > b.second;
                });

            // Assign population until we reach to the total population.
            int index = 0;
            while (assignedGenomes < remainingGenomes)
            {
                m_speciesData[residues[index++].first].m_population++;
                assignedGenomes++;
            }
        }
    }

    // Set remaining population. Remaining population is the total population of the species at the beginning of selection.
    for (auto& sData : m_speciesData)
    {
        sData.m_remainingPopulation = sData.m_population;
    }

    // Increment the current species index until the first species which has non-zero population.
    while (m_currentSpeciesDataIndex < (int)m_speciesData.size() && m_speciesData[m_currentSpeciesDataIndex].m_population == 0)
    {
        m_currentSpeciesDataIndex++;
    }

    // Calculate cumulative fitness of species if we will have inter species selections.
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
    assert(sData.m_remainingPopulation > 0);
    sData.m_remainingPopulation--;

    // Increment the current species index until the next species which has non-zero population.
    if (sData.m_remainingPopulation == 0)
    {
        do
        {
            m_currentSpeciesDataIndex++;
        } while (m_currentSpeciesDataIndex < (int)m_speciesData.size() && m_speciesData[m_currentSpeciesDataIndex].m_population == 0);
    }
}

bool SpeciesBasedGenomeSelector::preSelection(int numGenomesToSelect, SelectionMode mode)
{
    m_mode = mode;
    distributeSpeciesPopulations(numGenomesToSelect);
    return m_mode == GenomeSelector::SELECT_TWO_GENOMES ? m_numGenomes > 1 : m_numGenomes > 0;
}

bool SpeciesBasedGenomeSelector::postSelection()
{
#ifdef _DEBUG
    if(m_numGenomes != 0)
    {
        for (const auto& sData : m_speciesData)
        {
            assert(sData.m_remainingPopulation == 0);
        }
    }
#endif

    return true;
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

    // NOTE: std::uniform_real_distribution should return [min, max), but if we call randomReal(m_sumFitness[start], m_sumFitness[end])
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

        const SpeciesData& sData = m_speciesData[m_currentSpeciesDataIndex];

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
            // Keep selecting until we have different genomes.
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
        assert(m_speciesData.size() > 1);

        int selectedSpeciesId1, selectedSpeciesId2;

        auto select = [this](int& selectedSpeciesId)->const GenomeData*
        {
            // NOTE: std::uniform_real_distribution should return [min, max), but if we call randomReal(m_sumFitness[start], m_sumFitness[end])
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
                            selectedSpeciesId = i;
                            return sData.m_genomes[j];
                        }
                    }
                }
            }
            assert(0);
            return nullptr;
        };

        g1 = select(selectedSpeciesId1);
        selectedSpeciesId2 = selectedSpeciesId1;
        // Keep selecting until we have different species.
        while (selectedSpeciesId1 == selectedSpeciesId2)
        {
            g2 = select(selectedSpeciesId2);
        }
    }
}
