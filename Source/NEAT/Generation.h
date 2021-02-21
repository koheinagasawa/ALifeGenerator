/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenerationBase.h>
#include <NEAT/Genome.h>
#include <NEAT/Species.h>
#include <NEAT/DefaultMutation.h>
#include <NEAT/DefaultCrossOver.h>

namespace NEAT
{
    // Generation for NEAT.
    class Generation : public GenerationBase
    {
    public:
        using GenomePtr = std::shared_ptr<Genome>;
        using Genomes = std::vector<GenomePtr>;
        using SpeciesPtr = std::shared_ptr<Species>;
        using SpeciesList = std::unordered_map<SpeciesId, SpeciesPtr>;

        struct GenerationParams
        {
            // Minimum numbers of species members to copy its champion without modifying it.
            uint16_t m_minMembersInSpeciesToCopyChampion = 5;

            // Maximum count of generations which one species can stay in stagnant.
            // Species who is stagnant more than this count is not allowed to reproduce.
            uint16_t m_maxStagnantCount = 5;

            // Parameters used for distance calculation of two genomes.
            Genome::CalcDistParams m_calcDistParams;

            // Distance threshold used for speciation.
            float m_speciationDistanceThreshold = 3.f;
        };

        struct Cinfo
        {
            // The number of genomes in one generation.
            uint16_t m_numGenomes;

            // Cinfo for initial set genomes.
            Genome::Cinfo m_genomeCinfo;

            // Minimum weight for initial set of genomes.
            float m_minWeight = -1.f;

            // Maximum weight for initial set of genomes.
            float m_maxWeight = 1.f;

            // Fitness calculator.
            FitnessCalcPtr m_fitnessCalculator;

            // Parameters used for mutation.
            DefaultMutation::MutationParams m_mutationParams;

            // Parameters used for cross over.
            DefaultCrossOver::CrossOverParams m_crossOverParams;

            GenerationParams m_generationParams;

            // Random generator.
            PseudoRandom* m_random = nullptr;
        };

        // Constructor by Cinfo.
        Generation(const Cinfo& cinfo);

        // Constructor by a collection of Genomes.
        Generation(const Genomes& genomes, const Cinfo& cinfo);

        // Calculate fitness of all the genomes.
        void calcFitness();

        inline auto getGenomes() const->const GenomeDatas& { return *m_genomes; }

        inline auto getAllSpecies() const->const SpeciesList& { return m_species; }
        inline auto getSpecies(SpeciesId id) const->const SpeciesPtr { return m_species.find(id) != m_species.end() ? m_species.at(id) : nullptr; }

        auto getSpecies(GenomeId genomeId) const->SpeciesId;

        bool isSpeciesReproducible(SpeciesId speciesId) const;

    protected:
        void init(const Cinfo& cinfo);

        virtual void postUpdateGeneration() override;

        virtual auto createSelector()->GenomeSelectorPtr override;

        SpeciesList m_species;
        std::unordered_map<GenomeId, SpeciesId> m_genomesSpecies;
        UniqueIdCounter<SpeciesId> m_speciesIdGenerator;
        GenerationParams m_params;
    };
}
