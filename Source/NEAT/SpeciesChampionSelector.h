/*
* SpeciesChampionSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/Generation.h>

namespace NEAT
{
    class SpeciesChampionSelector : public GenomeGenerator
    {
    public:
        SpeciesChampionSelector(const Generation* g, float minMembersInSpeciesToCopyChampion);

        virtual void generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelectorBase* genomeSelector) override;

    protected:
        const Generation* m_generation;

        float m_minMembersInSpeciesToCopyChampion;
    };
}
