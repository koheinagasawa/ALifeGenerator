/*
* GenomeGenerator.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/Base/GenomeBase.h>

// Base class which generates a set of new genomes from existing genomes.
class GenomeGenerator
{
public:
    // Type declarations
    using GenomeBasePtr = std::shared_ptr<GenomeBase>;
    using GenomeBasePtrs = std::vector<GenomeBasePtr>;

    // Generate a set of new genomes by using genomeSelector.
    // genomeSelector has to be already configured and available to select existing genomes.
    virtual void generate(int numTotalGenomes, int numRemaningGenomes, class GenomeSelector* genomeSelector) = 0;

    // Returns the number of newly generated genomes.
    inline int getNumGeneratedGenomes() const { return (int)m_generatedGenomes.size(); }

    // Return the set of newly generated genomes.
    inline auto getGeneratedGenomes() const->const GenomeBasePtrs { return m_generatedGenomes; }

protected:
    // The newly generated genomes
    GenomeBasePtrs m_generatedGenomes;
};
