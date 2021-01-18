/*
* Genome.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <memory>
#include <functional>

#include <NEAT/MutableNetwork.h>

DECLARE_ID(InnovationId, uint64_t);
DECLARE_ID(GenomeId);
DECLARE_ID(GenerationId);

namespace NEAT
{
    // Singleton class to control InnovationId.
    class InnovationCounter
    {
    public:
        InnovationCounter() = default;

        // Returns a new innovation id.
        InnovationId getNewInnovationId();

    protected:
        InnovationCounter(const InnovationCounter&) = delete;
        void operator=(const InnovationCounter&) = delete;

        InnovationId m_innovationCount = 0;
    };

    struct Activation
    {
        using Func = std::function<float(float)>;

        float activate(float value) const { return m_func(value); }

        std::string m_name;
        Func m_func;
    };

    struct MutationParams
    {

    };

    // Genome for NEAT
    class Genome
    {
    public:
        struct InnovationEntry
        {
            bool operator<(const InnovationEntry& ie) { return m_id < ie.m_id; }

            InnovationId m_id;
            EdgeId m_edgeId;
        };

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

        using Network = MutableNetwork<Node>;
        using NetworkPtr = std::shared_ptr<Network>;
        using InnovationEntries = std::vector<InnovationEntry>;

        struct Cinfo
        {
            uint16_t m_numInputNode;
            uint16_t m_numOutputNode;
        };

        // Constructor with cinfo. It will construct the minimum dimensional network where there is no hidden node and
        // all input nodes and output nodes are fully connected.
        Genome(const Cinfo& cinfo);

        void mutate(const MutationParams& params, InnovationCounter& innovIdCounter);

        // Get innovations of this network. Returned list of innovation entries is sorted by innovation id.
        inline auto getInnovations() const->const InnovationEntries& { return m_innovations; }

    protected:
        NetworkPtr m_network;
        InnovationEntries m_innovations; // A list of innovations sorted by innovation id.
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
