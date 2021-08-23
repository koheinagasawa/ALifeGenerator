/*
* GenerationCloner.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/Base/Generators/GenomeGenerator.h>

// GenomeGenerator which just copies selected genomes.
template <typename GenomeType>
class GenomeCloner : public GenomeGenerator
{
public:
    // Generate a set of new genomes by using genomeSelector.
    // genomeSelector has to be already configured and available to select existing genomes.
    virtual void generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelector* genomeSelector) override;
};

template <typename GenomeType>
void GenomeCloner<GenomeType>::generate(int /*numTotalGenomes*/, int numRemaningGenomes, GenomeSelector* genomeSelector)
{
    using GenomeData = GenerationBase::GenomeData;

    // Clear new genomes output.
    m_generatedGenomes.clear();

    if (numRemaningGenomes == 0)
    {
        return;
    }

    genomeSelector->preSelection(numRemaningGenomes, GenomeSelector::SELECT_ONE_GENOME);

    m_generatedGenomes.reserve(numRemaningGenomes);

    // Copy genomes
    for (int i = 0; i < numRemaningGenomes; i++)
    {
        const GenomeData* g = genomeSelector->selectGenome();
        GenomeBasePtr copy = std::make_shared<GenomeType>(*static_cast<const GenomeType*>(g->getGenome().get()));
        m_generatedGenomes.push_back(std::static_pointer_cast<GenomeBase>(copy));
    }

    genomeSelector->postSelection();
}
