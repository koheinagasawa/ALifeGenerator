/*
* SpeciesChampionSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/Base/Generators/GenomeGenerator.h>
#include <NEAT/GeneticAlgorithms/NEAT/Species.h>

namespace NEAT
{
    // GenomeSelector which selects the best genome (champion) in a species.
    class SpeciesChampionSelector : public GenomeGenerator
    {
    public:
        using SpeciesPtr = std::shared_ptr<Species>;
        using SpeciesList = std::unordered_map<SpeciesId, SpeciesPtr>;

        SpeciesChampionSelector(float minMembersInSpeciesToCopyChampion);

        inline void updateSpecies(const SpeciesList& species) { m_species = &species; }

        // Generate new genomes by copying the champion in major species without modifying them.
        virtual void generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelector* genomeSelector) override;

    protected:
        const SpeciesList* m_species; // The Species.

        // Minimum numbers of members in a species to copy its champion.
        float m_minMembersInSpeciesToCopyChampion;
    };
}
