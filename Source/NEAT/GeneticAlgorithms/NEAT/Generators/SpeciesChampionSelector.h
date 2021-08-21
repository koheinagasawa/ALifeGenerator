/*
* SpeciesChampionSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/Base/Generators/GenomeGenerator.h>
#include <NEAT/GeneticAlgorithms/NEAT/Species.h>

#include <limits>

namespace NEAT
{
    // GenomeSelector which selects the best genome (champion) in a species.
    class SpeciesChampionSelector : public GenomeGenerator
    {
    public:
        using SpeciesPtr = std::shared_ptr<Species>;
        using SpeciesList = std::unordered_map<SpeciesId, SpeciesPtr>;

        SpeciesChampionSelector(float minMembersInSpeciesToCopyChampion);

        // Update species and the best fitness
        void updateSpecies(const SpeciesList& species, float bestFitness = std::numeric_limits<float>::max());

        // Generate new genomes by copying the champion in major species without modifying them.
        virtual void generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelector* genomeSelector) override;

        // Returns true since species champions should be protected from further modifications.
        virtual bool shouldGenomesProtected() const { return true; }

    protected:
        const SpeciesList* m_species = nullptr;     // The Species.
        float m_bestFitness;                        // The best fitness of the generation.
        float m_minMembersInSpeciesToCopyChampion;  // Minimum numbers of members in a species to copy its champion.
    };
}
