/*
* CrossOverDelegate.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenomeGenerator.h>

// GenomeGenerator which creates one new genome by doing cross-over two existing genomes.
class CrossOverDelegate : public GenomeGenerator
{
public:
    // Create a new genome by performing cross-over between genome1 and genome2.
    // sameFitness must be set true when genome1 and genome2 have the same fitness.
    virtual auto crossOver(const GenomeBase& genome1, const GenomeBase& genome2, bool sameFitness)->GenomeBasePtr = 0;
};
