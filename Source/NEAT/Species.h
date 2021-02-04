/*
* Species.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/Genome.h>
#include <Common/PseudoRandom.h>

namespace NEAT
{
    // Species for NEAT.
    class Species
    {
    public:
        using GenomePtr = std::shared_ptr<Genome>;

        Species(const Genome& initialRepresentative);

        // This should be called before creating a new generation.
        // This function will select a new representative genome for this species and clear all existing members.
        void preNewGeneration(PseudoRandom* random = nullptr);

        // Try to add the given genome to this species based on distance from its representative genome.
        // Return true if the genome is added to this species and otherwise return false.
        bool tryAddGenome(GenomePtr genome, float distanceThreshold, const Genome::CalcDistParams& params);

        inline bool hasMember() const { return m_members.size() > 0; }

    protected:
        std::vector<GenomePtr> m_members;
        Genome m_representative;
    };
}
