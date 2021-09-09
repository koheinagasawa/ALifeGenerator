/*
* Species.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <EvoAlgo/GeneticAlgorithms/NEAT/Genome.h>
#include <Common/PseudoRandom.h>

DECLARE_ID(SpeciesId, uint16_t);

namespace NEAT
{
    // Species for NEAT.
    class Species
    {
    public:
        using CGenomePtr = std::shared_ptr<const Genome>;

        // Constructor with representative.
        Species(const Genome& initialRepresentative);

        // Constructor with the first member.
        Species(CGenomePtr initialMember, float fitness);

        // This should be called before creating a new generation.
        // This function will select a new representative genome for this species and clear all existing members.
        void preNewGeneration();

        // This should be called after creating a new generation.
        // This function will update stagnant count.
        void postNewGeneration(RandomGenerator* random = nullptr);

        // Try to add the given genome to this species based on distance from its representative genome.
        // Return true if the genome is added to this species and otherwise return false.
        bool tryAddGenome(CGenomePtr genome, float fitness, float distanceThreshold, const Genome::CalcDistParams& params);

        // Add a genome to this species without checking its distance from the representative.
        void addGenome(CGenomePtr genome, float fitness);

        // Return the best genome in this species.
        inline auto getBestGenome() const->CGenomePtr { return m_bestGenome; }

        // Return the best fitness of this species.
        inline float getBestFitness() const { return m_bestFitness; }

        // Return the number of genomes in this species.
        inline int getNumMembers() const { return m_members.size(); }

        // Return the count of stagnant generations.
        inline int getStagnantGenerationCount() const { return m_stagnantCount; }

        // Return member genomes.
        inline auto getMembers() const->const std::vector<CGenomePtr>& { return m_members; }

        // Set reproducibility of this species.
        inline void setReproducible(bool enable) { m_reproducible = enable; }

        // Return true if this species is reproducible.
        inline bool isReproducible() const { return m_reproducible; }

    protected:
        std::vector<CGenomePtr> m_members;  // The members of this Species.
        Genome m_representative;            // The representative of this Species.
        CGenomePtr m_bestGenome;            // The best genome in this Species in the current generation.
        int m_stagnantCount = 0;            // The number of consecutive generations where there was no improvement on fitness.
        float m_bestFitness = 0.f;          // The best fitness in this Species of the current generation.
        float m_previousBestFitness = 0.f;  // The best fitness in this Species of the previous generation.
        bool m_reproducible = true;         // True if this species can reproduce descendants in the next generation.
    };
}
