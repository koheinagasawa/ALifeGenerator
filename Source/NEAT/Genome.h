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
#include <Common/UniqueIdCounter.h>

DECLARE_ID(GenomeId);

namespace NEAT
{
    // Helper class to manager unique node id and innovation id (edge id).
    class InnovationCounter
    {
    public:
        InnovationCounter() = default;

        NodeId getNewNodeId() { return m_nodeIdCounter.getNewId(); }
        EdgeId getNewInnovationId() { return m_innovationIdCounter.getNewId(); }

        void reset() { m_nodeIdCounter.reset(); m_innovationIdCounter.reset(); }

    protected:
        InnovationCounter(const InnovationCounter&) = delete;
        void operator=(const InnovationCounter&) = delete;

        UniqueIdCounter<NodeId> m_nodeIdCounter;
        UniqueIdCounter<EdgeId> m_innovationIdCounter;
    };

    // Genome for NEAT
    class Genome
    {
    public:
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
            // Type of Node
            enum class Type
            {
                INPUT,
                HIDDEN,
                OUTPUT,
                NONE
            };

            // Default constructor. This is used only by container of Node in Network class and users shouldn't call it.
            // Use Node(Type type) instead.
            Node() = default;

            // Constructor with node type.
            Node(Type type);

            // Copy constructor
            Node(const Node& other) = default;

            virtual float getValue() const override;
            virtual void setValue(float value) override;

            Type getNodeType() const { return m_type; }

            void setActivation(const Activation* activation) { m_activation = activation; }
            const std::string& getActivationName() const { return m_activation->m_name; }

        protected:
            float m_value = 0.f;
            Type m_type = Type::NONE;
            const Activation* m_activation = nullptr;

            friend class Genome;
        };

        // Structure used for constructor.
        struct Cinfo
        {
            // The number of input nodes.
            uint16_t m_numInputNodes = 1;

            // The number of output nodes.
            uint16_t m_numOutputNodes = 1;

            // The innovation counter. This has to be shared between all the genomes in one NEAT evaluation process.
            InnovationCounter* m_innovIdCounter = nullptr;
        };

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
            float m_weightMutationNewValMin = -10.f;

            // Maximum value when an edge gets a new random weight by mutation.
            float m_weightMutationNewValMax = 10.f;

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

        // Structure to store information about newly added edges by mutate().
        struct MutationOut
        {
            struct NewEdgeInfo
            {
                NodeId m_sourceInNode;
                NodeId m_sourceOutNode;
                EdgeId m_newEdge;
            };

            void clear();

            static constexpr int NUM_NEW_EDGES = 3;
            NewEdgeInfo m_newEdges[NUM_NEW_EDGES];
            int m_numNodesAdded;
            int m_numEdgesAdded;
        };

        // Parameters used for crossOver().
        struct CrossOverParams
        {
            // Probability of disabling inherited edge when either parent's edge is disabled.
            float m_disablingEdgeRate = 0.75f;

            // Probability of selecting inherit edge from genome1 for matching edges.
            float m_matchingEdgeSelectionRate = 0.5f;

            // Pseudo random generator. It can be null.
            RandomGenerator* m_random = nullptr;
        };

        // Parameters used for distance calculation
        struct CalcDistParams
        {
            // Factor for the number of disjoint edges.
            float m_disjointFactor;

            // Factor for weight differences.
            float m_weightFactor;

            // The minimum number of edges to apply normalization for the disjoint edge distance.
            int m_edgeNormalizationThreshold = 20;
        };

        // Type declaration
        using Network = MutableNetwork<Node>;
        using NetworkPtr = std::shared_ptr<Network>;

        // Constructor with cinfo. It will construct the minimum dimensional network where there is no hidden node and
        // all input nodes and output nodes are fully connected.
        Genome(const Cinfo& cinfo);

        // Copy constructor and operator
        Genome(const Genome& other);
        void operator= (const Genome& other);

        auto getNetwork() const->const Network* { return m_network.get(); }

        // Set weight of edge.
        void setEdgeWeight(EdgeId edgeId, float weight) { m_network->setWeight(edgeId, weight); }

        // Mutate this genome. There are three ways of mutation.
        // 1. Change weights of edges with a small perturbation.
        // 2. Add a new node at a random edge.
        // 3. Connect random two nodes by a new edge.
        // Probability of mutation and other parameters are controlled by MutationParams. See its comments for more details.
        void mutate(const MutationParams& params, MutationOut& mutationOut);

        // Cross over two genomes and generate a new one.
        // genome1 has to have higher fitting score.
        // Set sameFittingScore true if the fitting scores of genome1 and genome2 is the same.
        static Genome crossOver(const Genome& genome1, const Genome& genome2, bool sameFittingScore, const CrossOverParams& params);

        // Get innovations of this network. Returned list of innovation entries is sorted by innovation id.
        inline auto getInnovations() const->const Network::EdgeIds& { return m_innovations; }

        // Calculate and return distance between two genomes
        static float calcDistance(const Genome& genome1, const Genome& genome2, const CalcDistParams& params);

        // Return false if this genome contains any invalid data.
        bool validate() const;

    protected:
        // Constructor used by crossOver().
        Genome(InnovationCounter& innovationCounter);

        NetworkPtr m_network;                   // The network.
        Network::EdgeIds m_innovations;         // A list of innovations sorted by innovation id.
        InnovationCounter& m_innovIdCounter;    // The innovation counter shared by all the genomes.
    };
}
