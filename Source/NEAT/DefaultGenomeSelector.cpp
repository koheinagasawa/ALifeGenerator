/*
* DefaultGenomeSelector.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/DefaultGenomeSelector.h>

using namespace NEAT;

DefaultGenomeSelector::DefaultGenomeSelector(const Generation* generation, PseudoRandom& random)
    : GenomeSelectorBase(random)
    , m_generation(generation)
{

}

bool DefaultGenomeSelector::setGenomes(const GenomeDatas& genomesIn)
{
    assert(m_generation);
    assert(genomesIn.size() > 0);
    assert(genomesIn.size() == m_generation->getNumGenomes());

    m_genomes.reserve(genomesIn.size());
    m_sumFitness.reserve(genomesIn.size() + 1);
    float sumFitness = 0;
    m_sumFitness.push_back(0);

    const Generation::SpeciesList& species = m_generation->getAllSpecies();

    auto calcFitnessSharingFactor = [&species](SpeciesId speciesId)->float
    {
        return speciesId.isValid() ? 1.f / (float)species.at(speciesId)->getNumMembers() : 1.0f;
    };

    SpeciesId currentSpecies = getSpeciesId(genomesIn[0]);
    float fitnessSharingFactor = calcFitnessSharingFactor(currentSpecies);
    int currentSpeciesStartIndex = 0;

#ifdef _DEBUG
    std::unordered_set<SpeciesId> speciesIndices;
    speciesIndices.insert(currentSpecies);

    // Here we are assuming that genomes are sorted by species id.
    {
        SpeciesId curId = getSpeciesId(genomesIn[0]);
        for (const GenomeData& g : genomesIn)
        {
            if (!isGenomeReproducible(g) || g.getFitness() == 0)
            {
                continue;
            }

            SpeciesId id = getSpeciesId(g);
            if (curId != id)
            {
                assert(curId < id);
                curId = id;
            }
        }
    }
#endif

    for (const GenomeData& g : genomesIn)
    {
        if (!isGenomeReproducible(g) || g.getFitness() == 0)
        {
            continue;
        }

        if (currentSpecies != getSpeciesId(g))
        {
            m_spciecesStartEndIndices.insert({ currentSpecies, { currentSpeciesStartIndex, (int)m_genomes.size() } });

            currentSpecies = getSpeciesId(g);
            fitnessSharingFactor = calcFitnessSharingFactor(currentSpecies);
            currentSpeciesStartIndex = (int)m_genomes.size();
#ifdef _DEBUG
            assert(speciesIndices.find(currentSpecies) == speciesIndices.end());
#endif
        }

        m_genomes.push_back(&g);

        assert(g.getFitness() > 0);

        const float adjustedFitness = g.getFitness() * fitnessSharingFactor;
        sumFitness += adjustedFitness;
        m_sumFitness.push_back(sumFitness);
    }

    if (m_genomes.size() == 0)
    {
        return false;
    }

    if (sumFitness == 0.f)
    {
        // All genomes have 0 fitness.
        // Set up homogeneous distribution.
        for (int i = 0; i <= (int)m_genomes.size(); i++)
        {
            m_sumFitness[i] = (float)i;
        }
    }

    m_spciecesStartEndIndices.insert({ currentSpecies, { currentSpeciesStartIndex, (int)m_genomes.size() } });

    assert(m_genomes.size() + 1 == m_sumFitness.size());

    return true;
}

auto DefaultGenomeSelector::selectGenome()->const GenomeData* 
{
    return selectGenome(0, m_genomes.size());
}

void DefaultGenomeSelector::selectTwoGenomes(const GenomeData*& g1, const GenomeData*& g2)
{
    assert(m_spciecesStartEndIndices.size() > 0);

    g1 = nullptr;
    g2 = nullptr;

    // Select a random genome.
    g1 = selectGenome();
    assert(g1);

    // Get start and end indices of the species of g1.
    const IndexSet& startEnd = m_spciecesStartEndIndices.at(getSpeciesId(*g1));

    if (m_random.randomReal01() < m_interSpeciesCrossOverRate || (startEnd.m_end - startEnd.m_start) < 2)
    {
        // Inter species cross-over. Just select another genome among the entire generation.
        g2 = g1;
        while (g1 == g2)
        {
            g2 = selectGenome();
        }
    }
    else
    {
        // Intra species cross-over. Select another genome within the same species.

        if (startEnd.m_end - startEnd.m_start == 2)
        {
            // There are only two genomes in this species.
            g1 = m_genomes[startEnd.m_start];
            g2 = m_genomes[startEnd.m_end - 1];
        }
        else
        {
            // Select g2 among the species.
            g2 = g1;
            while (g1 == g2)
            {
                g2 = selectGenome(startEnd.m_start, startEnd.m_end);
            }
        }

        assert(getSpeciesId(*g1) == getSpeciesId(*g2));
    }
}

auto DefaultGenomeSelector::selectGenome(int start, int end)->const GenomeData*
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

SpeciesId DefaultGenomeSelector::getSpeciesId(const GenomeData& gd) const
{
    return m_generation->getSpecies(gd.getId());
}

bool DefaultGenomeSelector::isGenomeReproducible(const GenomeData& gd) const
{
    return !m_skipStagnantSpecies || m_generation->isSpeciesReproducible(getSpeciesId(gd));
}
