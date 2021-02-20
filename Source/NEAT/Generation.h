/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenerationBase.h>
#include <NEAT/Genome.h>
#include <NEAT/Species.h>

namespace NEAT
{
    class FitnessCalculatorBase
    {
    public:
        virtual float calcFitness(const Genome& genome) const = 0;
    };

    // Generation for NEAT.
    class Generation
    {
    public:
        using GenomePtr = std::shared_ptr<Genome>;
        using Genomes = std::vector<GenomePtr>;
        using SpeciesPtr = std::shared_ptr<Species>;
        using SpeciesList = std::unordered_map<SpeciesId, SpeciesPtr>;

        struct GenomeData
        {
        public:
            // Default constructor
            GenomeData() = default;

            inline auto getGenome() const->const Genome* { return m_genome.get(); }
            inline float getFitness() const { return m_fitness; }
            inline auto getSpeciesId() const->SpeciesId { return m_speciesId; }
            inline bool canReproduce() const { return m_canReproduce; }

        protected:
            // Constructor with a pointer to the genome and its id.
            GenomeData(GenomePtr genome, GenomeId id);

            // Initialize by a pointer to the genome and its id.
            void init(GenomePtr genome, GenomeId id);

            GenomePtr m_genome;
            GenomeId m_id;
            float m_fitness = 0.f;
            SpeciesId m_speciesId = SpeciesId::invalid();
            bool m_canReproduce = true;

            friend class Generation;
        };

        using GenomeDatas = std::vector<GenomeData>;
        using GenomeDatasPtr = std::shared_ptr<GenomeDatas>;

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
            FitnessCalculatorBase* m_fitnessCalculator;

            // Random generator.
            PseudoRandom* m_random = nullptr;
        };

        // Parameters used in createNewGeneration()
        struct CreateNewGenParams
        {
            // Parameters used for mutation.
            Genome::MutationParams m_mutationParams;

            // Parameters used for cross over.
            Genome::CrossOverParams m_crossOverParams;

            // Parameters used for distance calculation of two genomes.
            Genome::CalcDistParams m_calcDistParams;

            // Minimum numbers of species members to copy its champion without modifying it.
            uint16_t m_minMembersInSpeciesToCopyChampion = 5;

            // Maximum count of generations which one species can stay in stagnant.
            // Species who is stagnant more than this count is not allowed to reproduce.
            uint16_t m_maxStagnantCount = 5;

            // Rate of the number of genomes to generate by cross over.
            float m_crossOverRate = 0.75f;

            // Rate of interspecies crossover.
            float m_interSpeciesCrossOverRate = 0.001f;

            // Distance threshold used for speciation.
            float m_speciationDistanceThreshold = 3.f;

            // Random generator.
            PseudoRandom* m_random = nullptr;
        };

        // Constructor by Cinfo.
        Generation(const Cinfo& cinfo);

        // Constructor by a collection of Genomes.
        Generation(const Genomes& genomes, FitnessCalculatorBase* fitnessCalculator);

        // Create a new generation.
        void createNewGeneration(const CreateNewGenParams& params);

        // Calculate fitness of all the genomes.
        void calcFitness();

        inline auto getGenomes() const->const GenomeDatas& { return *m_genomes; }
        inline int getNumGenomes() const { return m_numGenomes; }

        inline auto getAllSpecies() const->const SpeciesList& { return m_species; }
        inline auto getSpecies(SpeciesId id) const->const SpeciesPtr { return m_species.find(id) != m_species.end() ? m_species.at(id) : nullptr; }

        inline auto getFitnessCalculator() const->const FitnessCalculatorBase& { return *m_fitnessCalculator; }

        inline auto getId() const->GenerationId { return m_id; }

        auto getSpecies(GenomeId genomeId) const->SpeciesId;

        bool isReproducible(SpeciesId speciesId) const;

    protected:
        // Called inside createNewGeneration().
        void addGenome(GenomePtr genome);

        // Helper class to select a random genome by taking fitness into account.
        class GenomeSelector
        {
        public:
            // Constructor
            GenomeSelector(PseudoRandom& random) : m_random(random) {}

            // Set genomes to select and initialize internal data.
            bool setGenomes(const Generation::GenomeDatas& genomesIn, const Generation::SpeciesList& species);

            // Select a random genome.
            inline auto selectRandomGenome()->const Generation::GenomeData* { return selectRandomGenome(0, m_genomes.size()); }

            // Select two random genomes in the same species.
            void selectTwoRandomGenomes(float interSpeciesCrossOverRate, const Generation::GenomeData*& g1, const Generation::GenomeData*& g2);

        protected:
            // Select a random genome between start and end (not including end) in m_genomes array.
            auto selectRandomGenome(int start, int end)->const Generation::GenomeData*;

            struct IndexSet
            {
                int m_start, m_end;
            };

            std::vector<const Generation::GenomeData*> m_genomes;
            std::vector<float> m_sumFitness;
            std::unordered_map<SpeciesId, IndexSet> m_spciecesStartEndIndices;
            PseudoRandom& m_random;
        };

        GenomeDatasPtr m_genomes;
        GenomeDatasPtr m_prevGenGenomes;
        SpeciesList m_species;
        UniqueIdCounter<SpeciesId> m_speciesIdGenerator;
        FitnessCalculatorBase* m_fitnessCalculator;
        int m_numGenomes;
        GenerationId m_id;
    };
}
