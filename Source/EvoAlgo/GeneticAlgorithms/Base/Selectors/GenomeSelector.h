/*
* GenomeSelectorBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <EvoAlgo/GeneticAlgorithms/Base/GenerationBase.h>

// Abstract class to select genomes.
class GenomeSelector
{
public:
    // Type declarations.
    using GenomeDatas = GenerationBase::GenomeDatas;
    using GenomeData = GenerationBase::GenomeData;
    
    enum SelectionMode
    {
        NONE,               // Invalid mode
        SELECT_ONE_GENOME,  // Select one genome
        SELECT_TWO_GENOMES  // Select two different genomes at once
    };

    // This function should be called before the first selection.
    virtual bool preSelection(int numGenomesToSelect, SelectionMode mode = NONE) { return true; }

    // This function should be called after the last selection.
    virtual bool postSelection() { return true; }

    // Select a random genome.
    virtual auto selectGenome()->const GenomeData* = 0;

    // Select two random genomes.
    virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) = 0;
};
