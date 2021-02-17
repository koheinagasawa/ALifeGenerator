/*
* DefaultMutation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenerationBase.h>
#include <Common/PseudoRandom.h>

namespace NEAT
{
    class DefaultMutation : public MutationDelegate
    {
    public:
        // Structure used for mutate().
        struct MutationParams
        {
            // Probability of weight mutation. It has to be between 0 and 1.
            float m_weightMutationRate = 0.8f;

            // Perturbation of weight mutation. It has to be between 0 and 1.
            // Mutated weight can be from [original * (1 - intensity)] to [original * (1 + intensity)].
            float m_weightMutationPerturbation = 0.05f;

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

            // Minimum weight for a new edge.
            float m_newEdgeMinWeight = -0.5f;

            // Maximum weight for a new edge.
            float m_newEdgeMaxWeight = 0.5f;

            // Pseudo random generator. It can be null.
            RandomGenerator* m_random = nullptr;
        };

        // Mutate this genome. There are three ways of mutation.
        // 1. Change weights of edges with a small perturbation.
        // 2. Add a new node at a random edge.
        // 3. Connect random two nodes by a new edge.
        // Probability of mutation and other parameters are controlled by MutationParams. See its comments for more details.
        virtual void mutate(GenomeBase* genomeIn, MutationOut& mutationOut) override;

        virtual auto mutate(const GenerationBase::GenomeDatas& generation, int numGenomesToMutate)->std::vector<std::shared_ptr<GenerationBase>> override;

        MutationParams m_params;
    };
}
