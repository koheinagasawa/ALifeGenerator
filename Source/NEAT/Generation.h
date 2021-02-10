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
            inline float getFitness() const { return m_fitness; }

        protected:
            GenomePtr m_genome;
            GenomeId m_id;
            float m_fitness = 0.f;
            bool m_canReproduce;

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

            FitnessCalculator* m_fitnessCalculator;

            // Random generator.
            PseudoRandom* m_random = nullptr;
        };

        // Constructor by Cinfo.
        Generation(const Cinfo& cinfo);

        // Constructor by a collection of Genomes.
        Generation(const Genomes& genomes, FitnessCalculator* fitnessCalculator);

        // Parameters used in createNewGeneration()
        struct CreateNewGenParams
        {
            // Parameters used for mutation.
            Genome::MutationParams m_mutationParams;

            // Parameters used for cross over.
            Genome::CrossOverParams m_crossOverParams;

            // Parameters used for distance calculation of two genomes.
            Genome::CalcDistParams m_calcDistParams;

            // Distance threshold used for speciation.
            float m_speciationDistanceThreshold;
        };

        // Create a new generation.
        auto createNewGeneration(const CreateNewGenParams& params) const->Generation;

        // Set values of input nodes.
        // inputNodeValues has to be the same size as the number of input nodes and has to be sorted in the same order as them.
        void setInputNodeValues(const std::vector<float>& values);

        inline auto getGenomes() const->const GenomeDatas& { return *m_genomes; }
        inline int getNumGenomes() const { return m_genomes->size(); }

        inline auto getFitnessCalculator() const->const FitnessCalculator& { return *m_fitnessCalculator; }

        inline auto getId() const->GenerationId { return m_id; }

    protected:
        // Constructor used in createNewGeneration().
        Generation(GenerationId id, FitnessCalculator* fitnessCalculator);

        GenomeDatasPtr m_genomes;
        SpeciesList m_species;
        FitnessCalculator* m_fitnessCalculator;
        GenerationId m_id;
    };
}
