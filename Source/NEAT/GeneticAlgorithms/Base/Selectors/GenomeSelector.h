/*
* GenomeSelectorBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/Base/GenerationBase.h>

// Abstract class to select genomes.
class GenomeSelector
{
public:
    // Type declarations.
    using GenomeDatas = GenerationBase::GenomeDatas;
    using GenomeData = GenerationBase::GenomeData;

    enum SelectionMode
    {
        ONE,
        TWO
    };

    // This function should be called before the first selection.
    virtual void preSelection(int numGenomesToSelect, SelectionMode mode) {}

    // This function should be called after the last selection.
    virtual void postSelection() {}

    // Select a random genome.
    virtual auto selectGenome()->const GenomeData* = 0;

    // Select two random genomes.
    virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) = 0;
};
