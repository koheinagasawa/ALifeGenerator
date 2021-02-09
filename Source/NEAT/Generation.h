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
    // Generation for NEAT.
    class Generation
    {
    public:
        using GenomePtr = std::shared_ptr<Genome>;
        using GenerationPtr = std::shared_ptr<Generation>;

        struct GenomeData
        {
        public:
            inline void setFitness(float fitness) { m_fitness = fitness; }
            inline float getFitness() const { return m_fitness; }

        protected:
            GenomePtr m_genome;
            GenomeId m_id;
            float m_fitness;

            friend class Generation;
        };

        using Genomes = std::vector<GenomeData>;
        using GenomesPtr = std::shared_ptr<Genomes>;

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

            // Random generator.
            PseudoRandom* m_random = nullptr;
        };

        // Constructor by Cinfo.
        Generation(const Cinfo& cinfo);

        // Constructor by a collection of Genomes.
        Generation(const GenomesPtr& genomes);

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
        GenerationPtr createNewGeneration(const CreateNewGenParams& params) const;

        inline auto getGenomes() const->const GenomesPtr& { return m_genomes; }

        inline auto getId() const->GenerationId { return m_id; }

    protected:
        // Constructor used in createNewGeneration().
        Generation(GenerationId id);

        GenomesPtr m_genomes;
        GenerationId m_id;
    };
}
