/*
* DefaultMutation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/PseudoRandom.h>
#include <NEAT/GeneticAlgorithms/NEAT/Genome.h>
#include <NEAT/GeneticAlgorithms/Base/Modifiers/MutationDelegate.h>

namespace NEAT
{
    // Default mutator class for NEAT.
    class DefaultMutation : public MutationDelegate
    {
    public:
        using GenomePtr = std::shared_ptr<Genome>;

        // Structure used for mutate().
        struct MutationParams
        {
            // Probability of weight mutation. It has to be between 0 and 1.
            float m_weightMutationRate = 0.8f;

            // Perturbation of weight mutation. It has to be between 0 and 1.
            // Mutated weight can be from [original * (1 - intensity)] to [original * (1 + intensity)].
            float m_weightMutationPerturbation = 0.2f;

            // Probability that an edge gets a new random weight instead of perturbation. It has to be between 0 and 1.
            float m_weightMutationNewValRate = 0.1f;

            // Minimum value when an edge gets a new random weight by mutation.
            float m_weightMutationValMin = -10.f;

            // Maximum value when an edge gets a new random weight by mutation.
            float m_weightMutationValMax = 10.f;

            // Probability of mutation to add a new node. It has to be between 0 and 1.
            float m_addNodeMutationRate = 0.03f;

            // Probability of mutation to add a new edge. It has to be between 0 and 1.
            float m_addEdgeMutationRate = 0.05f;

            // Probability of removing an existing edge.
            float m_removeEdgeMutationRate = 0.005f;

            // Minimum weight for a new edge.
            float m_newEdgeMinWeight = -10.f;

            // Maximum weight for a new edge.
            float m_newEdgeMaxWeight = 10.f;

            // Pseudo random generator. It can be null.
            RandomGenerator* m_random = nullptr;
        };

        // Constructors
        DefaultMutation() = default;
        DefaultMutation(const MutationParams& params) : m_params(params) {}

        void reset();

        // Mutate a single genome. There are four ways of mutation.
        // 1. Change weights of edges with a small perturbation.
        // 2. Remove a random existing edge.
        // 3. Add a new node at a random edge.
        // 4. Connect random two nodes by a new edge.
        // Probability of mutation and other parameters are controlled by MutationParams. See its comments for more details.
        virtual void mutate(GenomeBase* genomeInOut, MutationOut& mutationOut) override;

        // Modifies the genomes by mutation.
        // This functions appends the result of mutation to m_mutation.
        // If the applied mutation is identical to what is already stored in m_mutation, the mutation will be modified 
        // so that identical mutations have the same node/edge ids.
        virtual void modifyGenomes(GenomeBasePtr& genome) override;

    public: 
        // The parameter.
        MutationParams m_params;

    protected:
        std::vector<MutationOut> m_mutations;
    };
}
