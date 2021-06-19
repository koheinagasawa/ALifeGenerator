/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/Base/GenerationBase.h>
#include <NEAT/GeneticAlgorithms/NEAT/Genome.h>
#include <NEAT/GeneticAlgorithms/NEAT/Species.h>
#include <NEAT/GeneticAlgorithms/NEAT/Modifiers/DefaultMutation.h>
#include <NEAT/GeneticAlgorithms/NEAT/Generators/DefaultCrossOver.h>

namespace NEAT
{
    // Generation for NEAT.
    class Generation : public GenerationBase
    {
    public:
        // Type declarations.
        using GenomePtr = std::shared_ptr<Genome>;
        using Genomes = std::vector<GenomePtr>;
        using SpeciesPtr = std::shared_ptr<Species>;
        using SpeciesList = std::unordered_map<SpeciesId, SpeciesPtr>;
        using SpeciesChampionSelectorPtr = std::shared_ptr<class SpeciesChampionSelector>;
        using MutatorPtr = std::shared_ptr<class DefaultMutation>;

        // Parameters used for generation.
        struct GenerationParams
        {
            // Maximum count of generations which one species can stay in stagnant.
            // Species who is stagnant more than this count is not allowed to reproduce.
            uint16_t m_maxStagnantCount = 15;

            // Rate of interspecies crossover.
            float m_interSpeciesCrossOverRate = 0.001f;

            // Parameters used for distance calculation of two genomes.
            Genome::CalcDistParams m_calcDistParams;

            // Distance threshold used for speciation.
            float m_speciationDistanceThreshold = 3.f;
        };

        // Cinfo of generation.
        struct Cinfo
        {
            // The number of genomes in one generation.
            uint16_t m_numGenomes;

            // Cinfo for initial set genomes.
            Genome::Cinfo m_genomeCinfo;

            // Minimum weight for initial set of genomes.
            float m_minWeight = -10.f;

            // Maximum weight for initial set of genomes.
            float m_maxWeight = 10.f;

            // Fitness calculator.
            FitnessCalcPtr m_fitnessCalculator;

            // Parameters used for mutation.
            DefaultMutation::MutationParams m_mutationParams;

            // Parameters used for cross over.
            DefaultCrossOver::CrossOverParams m_crossOverParams;

            // Minimum numbers of species members to copy its champion without modifying it.
            uint16_t m_minMembersInSpeciesToCopyChampion = 5;

            // The generation params.
            GenerationParams m_generationParams;

            // Random generator.
            RandomGenerator* m_random = nullptr;
        };

        // Constructor by Cinfo.
        Generation(const Cinfo& cinfo);

        // Constructor by a collection of Genomes.
        Generation(const Genomes& genomes, const Cinfo& cinfo);

        // Returns the list of all genomes. Genomes are sorted by SpeciesId.
        inline auto getGenomes() const->const GenomeDatas& { return *m_genomes; }

        // Returns the list of all genomes in the order of fitness. The first genome is the best genome in this generation.
        auto getGenomesInFitnessOrder() const->GenomeDatas;

        // Returns the list of all species.
        inline auto getAllSpecies() const->const SpeciesList& { return m_species; }

        // Returns the list of all species in the order of the best fitness.
        auto getAllSpeciesInBestFitnessOrder() const->std::vector<SpeciesPtr>;

        // Returns pointer to a species.
        inline auto getSpecies(SpeciesId id) const->const SpeciesPtr { return (m_species.find(id) != m_species.end()) ? m_species.at(id) : nullptr; }

        // Returns SpeciesId of the genome.
        inline auto getSpecies(GenomeId genomeId) const->SpeciesId { return (m_genomesSpecies.find(genomeId) != m_genomesSpecies.end()) ? m_genomesSpecies.at(genomeId) : SpeciesId::invalid(); }

        // Returns true if the species can reproduce descendants to the next generation.
        bool isSpeciesReproducible(SpeciesId speciesId) const;

    protected:
        void init(const Cinfo& cinfo);

        virtual void preUpdateGeneration() override;
        virtual void postUpdateGeneration() override;

        virtual auto createSelector()->GenomeSelectorPtr override;

    public:
        GenerationParams m_params;                                  // The parameters for this generation.

    protected:
        SpeciesList m_species;                                      // The list of Species.
        std::unordered_map<GenomeId, SpeciesId> m_genomesSpecies;   // A map between genome and species.
        UniqueIdCounter<SpeciesId> m_speciesIdGenerator;            // Id generator for species.
        SpeciesChampionSelectorPtr m_speciesChampSelector;          // Generator to select species champion.
        MutatorPtr m_mutator;                                       // Genome mutator.
    };
}
