/*
* HomogeneousGenomeSelector.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/HomogeneousGenomeSelector.h>

HomogeneousGenomeSelector::HomogeneousGenomeSelector(PseudoRandom& random)
    : GenomeSelectorBase(random)
{
}

bool HomogeneousGenomeSelector::setGenomes(const GenomeDatas& genomes)
{
    m_genomes = &genomes;
    return true;
}

auto HomogeneousGenomeSelector::selectGenome()->const GenomeData*
{
    assert(m_genomes);
    return &(*m_genomes)[m_random.randomInteger(0, (int)m_genomes->size() - 1)];
}

void HomogeneousGenomeSelector::selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2)
{
    assert(m_genomes);
    assert(m_genomes->size() > 1);

    genome1 = selectGenome();
    genome2 = genome1;

    while (genome1 == genome2)
    {
        genome2 = selectGenome();
    }
}
