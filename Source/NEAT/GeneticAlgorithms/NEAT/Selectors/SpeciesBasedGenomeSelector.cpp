/*
* SpeciesBasedGenomeSelector.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Selectors/SpeciesBasedGenomeSelector.h>

using namespace NEAT;

SpeciesBasedGenomeSelector::SpeciesBasedGenomeSelector(const GenomeDatas& genomeData, const SpeciesList& species, const GenomeSpeciesMap& genomeSpeciesMap, PseudoRandom* random)
    : GenomeSelector()
    , m_random(random ? *random : PseudoRandom::getInstance())
{
    const int numGenomes = (int)genomeData.size();
    assert(numGenomes > 0);

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

    m_speciesData.resize(species.size());
    m_speciesData.resize(1);

    m_totalFitness = 0;
    m_numGenomes = 0;
    m_hasSpeciesMoreThanOneMember = false;

    // Helper functions to calculate factor for fitness sharing.
    // Genome's fitness is going to be normalized by the number of members in its species.
    auto calcFitnessSharingFactor = [this](SpeciesPtr species)->float
    {
        return species ? 1.f / (float)species->getNumMembers() : 1.0f;
    };

    SpeciesId currentSpeciesId = SpeciesId::invalid();
    SpeciesPtr currentSpecies = nullptr;
    SpeciesData* speciesData;
    float fitnessSharingFactor;

    auto addGenomes = [&]()
    {
        // Calculate adjusted fitness for each genome and sum up those fitnesses.
        for (const GenomeData& g : genomeData)
        {
            const SpeciesId sId = genomeSpeciesMap.find(g.getId()) != genomeSpeciesMap.end() ? genomeSpeciesMap.at(g.getId()) : SpeciesId::invalid();
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

                m_speciesData.push_back(SpeciesData());
                speciesData = &m_speciesData.back();
                speciesData->m_species = currentSpecies;

                fitnessSharingFactor = currentSpecies ? 1.f / (float)currentSpecies->getNumMembers() : 1.0f;
            }
            else
            {
                m_hasSpeciesMoreThanOneMember = true;
            }

            float fitness = g.getFitness() * fitnessSharingFactor;
            speciesData->m_genomes.push_back(&g);
            speciesData->m_sumFitness += fitness;
            m_totalFitness += fitness;
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
}

void SpeciesBasedGenomeSelector::setNumGenomesToSelect(int numGenomes)
{

}

auto SpeciesBasedGenomeSelector::selectGenome()->const GenomeData* 
{
    return selectGenome(0, m_numGenomes);
}

void SpeciesBasedGenomeSelector::selectTwoGenomes(const GenomeData*& g1, const GenomeData*& g2)
{
    assert(m_spciecesStartEndIndices.size() > 0);
    assert(hasSpeciesMoreThanOneMember());

    g1 = nullptr;
    g2 = nullptr;

    if (m_genomes.size() < 2)
    {
        return;
    }

    const IndexSet* startEnd;

    while (1)
    {
        // Select a random genome.
        g1 = selectGenome();
        assert(g1);

        // Get start and end indices of the species of g1.
        startEnd = &m_spciecesStartEndIndices.at(getSpeciesId(*g1));

        // Check if this species has more than one member, otherwise try to select other species.
        if ((startEnd->m_end - startEnd->m_start) >= 2)
        {
            break;
        }
    }

    // Intra species cross-over. Select another genome within the same species.
    if (startEnd->m_end - startEnd->m_start == 2)
    {
        // There are only two genomes in this species.
        g1 = m_genomes[startEnd->m_start];
        g2 = m_genomes[startEnd->m_end - 1];
    }
    else
    {
        // Select g2 among the species.
        g2 = g1;
        while (g1 == g2)
        {
            g2 = selectGenome(startEnd->m_start, startEnd->m_end);
        }
    }

    assert(getSpeciesId(*g1) == getSpeciesId(*g2));
}

auto SpeciesBasedGenomeSelector::selectGenome(int start, int end)->const GenomeData*
{
    assert(m_genomes.size() > 0 && (m_genomes.size() + 1 == m_sumFitness.size()));
    assert(start >= 0 && end < (int)m_sumFitness.size() && start < end);

    if (m_sumFitness[start] < m_sumFitness[end])
    {
        // std::uniform_real_distribution should return [min, max), but if we call randomReal(m_sumFitness[start], m_sumFitness[end])
        // we see v == m_sumFitness[end] here for some reason. That's why we have to calculate nexttoward of max here to avoid unintentional
        // calculation later.
        const float v = m_random.randomReal(m_sumFitness[start], std::nexttoward(m_sumFitness[end], -1.f));
        for (int i = start; i < end; i++)
        {
            if (v < m_sumFitness[i + 1])
            {
                return m_genomes[i];
            }
        }

        assert(0);
        return nullptr;
    }
    else
    {
        // Fitness are all the same. Just select one by randomly.
        return m_genomes[m_random.randomInteger(start, end - 1)];
    }
}

auto SpeciesBasedGenomeSelector::getSpeciesId(const GenomeData& gd)->SpeciesId const
{
    return m_genomeSpeciesMap.find(gd.getId()) != m_genomeSpeciesMap.end() ? m_genomeSpeciesMap.at(gd.getId()) : SpeciesId::invalid();
}
