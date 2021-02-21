/*
* DefaultCrossOver.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/PseudoRandom.h>
#include <NEAT/Genome.h>
#include <NEAT/CrossOverDelegate.h>

namespace NEAT
{
    class DefaultCrossOver : public CrossOverDelegate
    {
    public:
        using GenomePtr = std::shared_ptr<Genome>;

        // Parameters used for crossOver().
        struct CrossOverParams
        {
            // Probability of disabling inherited edge when either parent's edge is disabled.
            float m_disablingEdgeRate = 0.75f;

            // Probability of selecting inherit edge from genome1 for matching edges.
            float m_matchingEdgeSelectionRate = 0.5f;

            // Rate of interspecies crossover.
            float m_interSpeciesCrossOverRate = 0.001f;

            // Pseudo random generator. It can be null.
            RandomGenerator* m_random = nullptr;
        };

        // Constructor
        DefaultCrossOver(const CrossOverParams& params) : m_params(params) {}

        // Cross over two genomes and generate a new one.
        // genome1 has to have higher fitting score.
        // Set sameFittingScore true if the fitting scores of genome1 and genome2 is the same.
        virtual auto crossOver(const GenomeBase& genome1, const GenomeBase& genome2, bool sameFitness)->GenomeBasePtr override;

        virtual void generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelectorBase* genomeSelector) override;

        CrossOverParams m_params;
    };
}
