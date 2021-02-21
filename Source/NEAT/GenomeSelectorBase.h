/*
* GenomeSelectorBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/PseudoRandom.h>
#include <NEAT/GenerationBase.h>

// Abstract class to select a genome.
class GenomeSelectorBase
{
public:
    using GenomeDatas = GenerationBase::GenomeDatas;
    using GenomeData = GenerationBase::GenomeData;

    // Constructor
    GenomeSelectorBase(PseudoRandom& random) : m_random(random) {}

    // Set genomes to select and initialize internal data.
    virtual bool setGenomes(const GenomeDatas& generation) = 0;

    // Select a random genome.
    virtual auto selectGenome()->const GenomeData* = 0;

    // Select two random genomes.
    virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) = 0;

protected:
    PseudoRandom& m_random;
};
