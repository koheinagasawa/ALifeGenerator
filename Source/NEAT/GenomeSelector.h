/*
* GenomeSelectorBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenerationBase.h>

// Abstract class to select genomes.
class GenomeSelector
{
public:
    // Type declarations.
    using GenomeDatas = GenerationBase::GenomeDatas;
    using GenomeData = GenerationBase::GenomeData;

    // Set genomes to select and initialize internal data.
    virtual bool setGenomes(const GenomeDatas& genomes) = 0;

    // Select a random genome.
    virtual auto selectGenome()->const GenomeData* = 0;

    // Select two random genomes.
    virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) = 0;
};
