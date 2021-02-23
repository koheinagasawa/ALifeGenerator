/*
* Genome.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenomeBase.h>
#include <Common/UniqueIdCounter.h>

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
    class Genome : public GenomeBase
    {
    public:
        // Struct used for constructor.
        struct Cinfo
        {
            // The number of input nodes.
            uint16_t m_numInputNodes = 1;

            // The number of output nodes.
            uint16_t m_numOutputNodes = 1;

            // The innovation counter. This has to be shared between all the genomes in one NEAT evaluation process.
            InnovationCounter* m_innovIdCounter = nullptr;

            // Default activation functions used during evaluation at each node.
            // If it's nullptr, input values are merely passed as an output of the node.
            const Activation* m_defaultActivation = nullptr;
        };

        // Parameters used to calculation distance between two genomes.
        struct CalcDistParams
        {
            // Factor for the number of disjoint edges.
            float m_disjointFactor = 1.f;

            // Factor for weight differences.
            float m_weightFactor = 0.4f;

            // The minimum number of edges to apply normalization for the disjoint edge distance.
            int m_edgeNormalizationThreshold = 20;
        };

        // Constructor with cinfo. It will construct the minimum dimensional network where there is no hidden node and
        // all input nodes and output nodes are fully connected.
        Genome(const Cinfo& cinfo);

        // Constructor with existing network and an offspring genome. This should be used by CrossOverDelegate.
        Genome(const Genome& other, NetworkPtr network, const Network::EdgeIds& innovations);

        // Copy constructor and operator
        Genome(const Genome& other);
        void operator= (const Genome& other);

        //
        // Network modification
        //

        // Add a new node by dividing the edge at edgeId.
        void addNodeAt(EdgeId edgeId, NodeId& newNode, EdgeId& newIncomingEdge, EdgeId& newOutgoingEdge);

        // Add a new edge between inNode and outNode with weight.
        EdgeId addEdgeAt(NodeId inNode, NodeId outNode, float weight);

        //
        // Innovation interface
        //

        // Get innovations of this network. Returned list of innovation entries is sorted by innovation id.
        inline auto getInnovations() const->const Network::EdgeIds& { return m_innovations; }

        // Reassign innovation id to an existing edge.
        // This functionality is used when there is the same structural mutation in more than one genomes in the same generation.
        void reassignInnovation(const EdgeId originalId, const EdgeId newId);

        // Calculate and return distance between two genomes
        static float calcDistance(const Genome& genome1, const Genome& genome2, const CalcDistParams& params);

        // Return false if this genome contains any invalid data.
        bool validate() const;

    protected:
        Network::EdgeIds m_innovations;         // A list of innovations sorted by innovation id.
        InnovationCounter& m_innovIdCounter;    // The innovation counter shared by all the genomes.
    };
}
