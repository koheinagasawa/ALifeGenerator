/*
* Genome.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <memory>
#include <functional>

#include <NEAT/MutableNetwork.h>
#include <Common/PseudoRandom.h>

DECLARE_ID(InnovationId, uint64_t);
DECLARE_ID(GenomeId);
DECLARE_ID(GenerationId);

namespace NEAT
{
    // Helper class to increment innovation id.
    class InnovationCounter
    {
    public:
        InnovationCounter() = default;

        // Returns a new innovation id. New id will be returned every time you call this function.
        InnovationId getNewInnovationId();

    protected:
        InnovationCounter(const InnovationCounter&) = delete;
        void operator=(const InnovationCounter&) = delete;

        InnovationId m_innovationCount = 0;
    };

    // Genome for NEAT
    class Genome
    {
    public:
        // Data structure which associates each edge in this genome and its innovation id.
        struct InnovationEntry
        {
            bool operator<(const InnovationEntry& ie) { return m_id < ie.m_id; }

            InnovationId m_id;
            EdgeId m_edgeId;
        };

        // Wrapper struct for activation function.
        struct Activation
        {
            using Func = std::function<float(float)>;

            float activate(float value) const { return m_func(value); }

            std::string m_name;
            Func m_func;
        };

        // Node structure of the genome.
        struct Node : public NodeBase
        {
        public:
            virtual float getValue() const override;
            virtual void setValue(float value) override;

            void setActivation(const Activation* activation) { m_activation = activation; }
            const std::string& getActivationName() const { return m_activation->m_name; }

        protected:
            float m_value;
            const Activation* m_activation;
        };

        // Structure used for constructor.
        struct Cinfo
        {
            // The number of input nodes.
            uint16_t m_numInputNode;

            // The number of output nodes.
            uint16_t m_numOutputNode;

            // The innovation counter. This has to be shared between all the genomes in one NEAT evaluation process.
            InnovationCounter& m_innovIdCounter;
        };

        // Structure used for mutate().
        struct MutationParams
        {
            // Probability of weight mutation. It has to be between 0 and 1.
            float m_weightMutationRate;

            // Intensity of weight mutation. It has to be between 0 and 1.
            // Mutated weight can be from [original * (1 - intensity)] to [original * (1 + intensity)].
            float m_weightMutationIntensity;

            // Pseudo random generator. It can be null.
            PseudoRandom* m_random = nullptr;
        };

        // Type declaration
        using Network = MutableNetwork<Node>;
        using NetworkPtr = std::shared_ptr<Network>;
        using InnovationEntries = std::vector<InnovationEntry>;

        // Constructor with cinfo. It will construct the minimum dimensional network where there is no hidden node and
        // all input nodes and output nodes are fully connected.
        Genome(const Cinfo& cinfo);

        // Mutate this genome. There are three ways to mutate.
        // 1. Change weights of edges with a small perturbation.
        // 2. Add a new node at a random edge.
        // 3. Connect random two nodes by a new edge.
        // Probability of mutation and other parameters are controlled by MutationParams. See its comments for more details.
        void mutate(const MutationParams& params);

        // Get innovations of this network. Returned list of innovation entries is sorted by innovation id.
        inline auto getInnovations() const->const InnovationEntries& { return m_innovations; }

        // Cross over two genomes and generate a new one.
        static Genome crossOver(const Genome& genome1, const Genome& genome2);

    protected:
        // Constructor used by mutate() and crossOver().
        Genome(InnovationCounter& innovationCounter);

        NetworkPtr m_network;                   // The network.
        InnovationEntries m_innovations;        // A list of innovations sorted by innovation id.
        InnovationCounter& m_innovIdCounter;    // The innovation counter shared by all the genomes.
    };

    class Generation
    {
    public:
        using GenomePtr = std::shared_ptr<const Genome>;

        struct GenomeData
        {
        public:
            void setFitness(float fitness) const { m_fitness = fitness; }

        protected:
            GenomePtr m_genome;
            GenomeId m_id;
            mutable float m_fitness;
        };

        using Genomes = std::shared_ptr<GenomeData>;
        using GenerationPtr = std::shared_ptr<Generation>;

        GenerationPtr createNewGeneration() const;

        const Genomes& getGenomes();

    protected:
        Genomes m_genomes;
        GenerationId m_id;
    };

    class NEAT
    {
    public:
        using GenerationPtr = Generation::GenerationPtr;

        void createNewGeneration();

        const Generation& getCurrentGeneration() const;

        void calcFitnessOfCurrentGen();
        float getMaxFitnessInCurrentGen() const;

    protected:
        virtual float calcFitness(const Genome& genome) const = 0;

        GenerationPtr m_currentGen;
        GenerationPtr m_previousGen;
    };
}
