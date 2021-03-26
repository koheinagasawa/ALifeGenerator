/*
* TestUtils.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/NEAT/Genome.h>

namespace TestUtils
{
    // Helper function to compare two genomes' structure.
    // Returns true if the two genomes have the same structure.
    bool compareGenome(const NEAT::Genome& g1, const NEAT::Genome& g2);

    // Helper function to compare two genomes' structure and edge's weights and states.
    // Returns true if the two genomes have the same structure, weights and states.
    bool compareGenomeWithWeightsAndStates(const NEAT::Genome& g1, const NEAT::Genome& g2);
}

