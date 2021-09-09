/*
* UniformGenomeSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/PseudoRandom.h>
#include <EvoAlgo/GeneticAlgorithms/Base/Selectors/GenomeSelector.h>

// Genome selector which selects randomly and uniformly.
class UniformGenomeSelector : public GenomeSelector
{
public:
    // Constructor
    UniformGenomeSelector(const GenomeDatas& genomes, RandomGenerator* random = nullptr);

    // Select a random genome.
    virtual auto selectGenome()->const GenomeData* override;

    // Select two random genomes.
    virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override;

protected:
    const GenomeDatas* m_genomes = nullptr;
    RandomGenerator& m_random;
};
