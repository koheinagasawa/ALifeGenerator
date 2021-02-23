/*
* SpeciesChampionSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/Generation.h>

namespace NEAT
{
    // GenomeSelector which selects the best genome (champion) in a species.
    class SpeciesChampionSelector : public GenomeGenerator
    {
    public:
        SpeciesChampionSelector(const Generation* g, float minMembersInSpeciesToCopyChampion);

        // Generate new genomes by copying the champion in major species without modifying them.
        virtual void generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelector* genomeSelector) override;

    protected:
        const Generation* m_generation; // The generation.

        // Minimum numbers of members in a species to copy its champion.
        float m_minMembersInSpeciesToCopyChampion;
    };
}
