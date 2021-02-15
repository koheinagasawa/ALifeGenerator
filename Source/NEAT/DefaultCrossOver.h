/*
* DefaultCrossOver.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenerationBase.h>
#include <Common/PseudoRandom.h>

namespace NEAT
{
    class DefaultCrossOver : public CrossOverDelegate
    {
    public:
        // Parameters used for crossOver().
        struct CrossOverParams
        {
            // Probability of disabling inherited edge when either parent's edge is disabled.
            float m_disablingEdgeRate = 0.75f;

            // Probability of selecting inherit edge from genome1 for matching edges.
            float m_matchingEdgeSelectionRate = 0.5f;

            // Pseudo random generator. It can be null.
            RandomGenerator* m_random = nullptr;
        };

        // Cross over two genomes and generate a new one.
        // genome1 has to have higher fitting score.
        // Set sameFittingScore true if the fitting scores of genome1 and genome2 is the same.
        virtual auto crossOver(const GenomeBase& genome1, const GenomeBase& genome2, bool sameFitness)->GenomeBase* override;

        CrossOverParams m_params;
    };
}
