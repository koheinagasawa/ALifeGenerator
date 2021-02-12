/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/Genome.h>
#include <NEAT/Species.h>

DECLARE_ID(GenerationId);
DECLARE_ID(GenomeId);

namespace NEAT
{
    class FitnessCalculator
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
        using SpeciesList = std::vector<Species>;

        struct GenomeData
        {
        public:
            // Default constructor
            GenomeData() = default;
            // Constructor with a pointer to the genome and its id.
            GenomeData(GenomePtr genome, GenomeId id);

            // Initialize by a pointer to the genome and its id.
            void init(GenomePtr genome, GenomeId id);

            inline float getFitness() const { return m_fitness; }
            inline int getSpeciesIndex() const { return m_speciesIndex; }
            inline bool canReproduce() const { return m_canReproduce; }

        protected:
            GenomePtr m_genome;
            GenomeId m_id;
            float m_fitness = 0.f;
            int m_speciesIndex = -1;
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
            float m_minWeight;

            // Maximum weight for initial set of genomes.
            float m_maxWeight;

            // Fitness calculator.
            FitnessCalculator* m_fitnessCalculator;

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
            float m_speciationDistanceThreshold;

            // Random generator.
            PseudoRandom* m_random = nullptr;
        };

        // Constructor by Cinfo.
        Generation(const Cinfo& cinfo);

        // Constructor by a collection of Genomes.
        Generation(const Genomes& genomes, FitnessCalculator* fitnessCalculator);

        // Create a new generation.
        void createNewGeneration(const CreateNewGenParams& params);

        // Set values of input nodes.
        // inputNodeValues has to be the same size as the number of input nodes and has to be sorted in the same order as them.
        void setInputNodeValues(const std::vector<float>& values);

        inline auto getGenomes() const->const GenomeDatas& { return *m_genomes; }
        inline int getNumGenomes() const { return m_numGenomes; }

        inline auto getSpecies() const->const SpeciesList& { return m_species; }

        inline auto getFitnessCalculator() const->const FitnessCalculator& { return *m_fitnessCalculator; }

        inline auto getId() const->GenerationId { return m_id; }

    protected:
        void addGenome(GenomePtr genome);

        GenomeDatasPtr m_genomes;
        GenomeDatasPtr m_prevGenGenomes;
        SpeciesList m_species;
        FitnessCalculator* m_fitnessCalculator;
        GenerationId m_id;
        int m_numGenomes;
    };
}
