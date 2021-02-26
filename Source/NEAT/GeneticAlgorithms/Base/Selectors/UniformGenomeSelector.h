/*
* UniformGenomeSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/PseudoRandom.h>
#include <NEAT/GeneticAlgorithms/Base/Selectors/GenomeSelector.h>

// Genome selector which selects randomly and uniformly.
class UniformGenomeSelector : public GenomeSelector
{
public:
    // Constructor
    UniformGenomeSelector(const GenomeDatas& genomes, PseudoRandom* random = nullptr);

    // Select a random genome.
    virtual auto selectGenome()->const GenomeData* override;

    // Select two random genomes.
    virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override;

protected:
    const GenomeDatas* m_genomes = nullptr;
    PseudoRandom& m_random;
};