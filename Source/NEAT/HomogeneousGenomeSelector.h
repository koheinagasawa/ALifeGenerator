/*
* HomogeneousGenomeSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenomeSelector.h>

// Genome selector which selects randomly and uniformly.
class HomogeneousGenomeSelector : public GenomeSelector
{
public:
    // Constructor
    HomogeneousGenomeSelector(PseudoRandom& random);

    // Set genomes to select and initialize internal data.
    virtual bool setGenomes(const GenomeDatas& genomes) override;

    // Select a random genome.
    virtual auto selectGenome()->const GenomeData* override;

    // Select two random genomes.
    virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override;

protected:
    const GenomeDatas* m_genomes = nullptr;
};
