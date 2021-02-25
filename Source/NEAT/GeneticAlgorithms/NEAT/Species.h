/*
* Species.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/NEAT/Genome.h>
#include <Common/PseudoRandom.h>


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
        void preNewGeneration(RandomGenerator* random = nullptr);

        // This should be called after creating a new generation.
        // This function will update stagnant count.
        void postNewGeneration();

        // Try to add the given genome to this species based on distance from its representative genome.
        // Return true if the genome is added to this species and otherwise return false.
        bool tryAddGenome(CGenomePtr genome, float fitness, float distanceThreshold, const Genome::CalcDistParams& params);

        inline auto getBestGenome() const->CGenomePtr { return m_bestGenome; }

        inline int getNumMembers() const { return m_members.size(); }

        inline int getStagnantGenerationCount() const { return m_stagnantCount; }

        inline auto getMembers() const->const std::vector<CGenomePtr>& { return m_members; }

        inline void setReproducible(bool enable) { m_reproducible = enable; }

        inline bool isReproducible() const { return m_reproducible; }

    protected:
        std::vector<CGenomePtr> m_members;  // The members of this Species.
        Genome m_representative;            // The representative of this Species.
        CGenomePtr m_bestGenome;            // The best genome in this Species in the current generation.
        int m_stagnantCount = 0;            // The number of consecutive generations where there was no improvement on fitness.
        float m_bestFitness = 0.f;          // The best fitness in this Species of the current generation.
        float m_previousBestFitness = 0.f;  // The best fitness in this Species of the previous generation.
        bool m_reproducible = true;       // True if this species can reproduce descendants in the next generation.
    };
}
