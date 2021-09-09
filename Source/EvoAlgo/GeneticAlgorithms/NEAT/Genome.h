/*
* Genome.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <EvoAlgo/GeneticAlgorithms/Base/GenomeBase.h>
#include <Common/UniqueIdCounter.h>

class ActivationProvider;

namespace NEAT
{
    // Helper class to manager unique node id and innovation id (edge id).
    class InnovationCounter
    {
    public:
        // Innovation (edge) that has ever created.
        struct EdgeEntry
        {
            NodeId m_inNode;
            NodeId m_outNode;

            inline bool operator==(const EdgeEntry& other) const
            {
                return m_inNode == other.m_inNode && m_outNode == other.m_outNode;
            }
        };

        // The specialized hash function for EdgeEntry.
        struct EdgeEntryHash
        {
            inline std::size_t operator() (const InnovationCounter::EdgeEntry& entry) const
            {
                return (std::hash<NodeId>()(entry.m_inNode)) ^ ((std::hash<NodeId>()(entry.m_outNode)) << 1);
            }
        };

        // Default constructor.
        InnovationCounter() = default;

        // Return a new node id.
        inline NodeId getNewNodeId() { return m_nodeIdCounter.getNewId(); }

        // Return a edge id. If the entry has already created before, then it returns an id from the history. Otherwise, it returns a new id.
        EdgeId getEdgeId(const EdgeEntry& entry);

        // Reset the counter and history.
        void reset();

    protected:
        // Prohibit copying.
        InnovationCounter(const InnovationCounter&) = delete;
        void operator=(const InnovationCounter&) = delete;

        using HistroyMap = std::unordered_map<EdgeEntry, EdgeId, EdgeEntryHash>;

        UniqueIdCounter<NodeId> m_nodeIdCounter;        // Counter of node ids.
        UniqueIdCounter<EdgeId> m_innovationIdCounter;  // Counter of innovation (edge) ids.
        HistroyMap m_innovationHistory;                 // History of all the innovations that ever happened before.
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

            // True to create a bias node.
            bool m_createBiasNode = false;

            // Default value of bias node.
            float m_biasNodeValue = 1.0f;

            // The innovation counter. This has to be shared between all the genomes in one NEAT evaluation process.
            InnovationCounter* m_innovIdCounter = nullptr;

            // Activation provider for the initial network.
            // If it's nullptr, input values are merely passed as an output of the node.
            const ActivationProvider* m_activationProvider = nullptr;

            // Type of the network.
            NeuralNetworkType m_networkType = NeuralNetworkType::FEED_FORWARD;
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

        // Create a clone of this genome.
        virtual std::shared_ptr<GenomeBase> clone() const override;

        //
        // Network modification
        //

        // Add a new node by dividing the edge at edgeId.
        void addNodeAt(EdgeId edgeId, const Activation* activation, NodeId& newNode, EdgeId& newIncomingEdge, EdgeId& newOutgoingEdge);

        // Add a new edge between inNode and outNode with weight.
        // When tryAddFlippedEdgeOnFail is true, after failing to add the original edge due to circular network,
        // this function will try to add an edge by flipping inNode and outNode.
        EdgeId addEdgeAt(NodeId inNode, NodeId outNode, float weight, bool tryAddFlippedEdgeOnFail = true);

        // Remove an existing edge.
        void removeEdge(EdgeId edge);

        // Reassign node id to an existing node.
        // This functionality should be only used when there is the same structural mutation in more than one genomes in the same generation.
        void reassignNodeId(const NodeId originalId, const NodeId newId);

        // Reassign a new node id to an existing node.
        // This functionality should be only used by mutator for mutating activation of a node.
        void reassignNewNodeIdAndConnectedEdgeIds(const NodeId originalId);

        // Reassign an innovation id to an existing edge.
        // This functionality should be only used when there is the same structural mutation in more than one genomes in the same generation.
        void reassignInnovation(const EdgeId originalId, const EdgeId newId);

        //
        // Other functions
        //

        // Get innovations of this network. Returned list of innovation entries is sorted by innovation id.
        inline auto getInnovations() const->const Network::EdgeIds& { return m_innovations; }

        // Return false if this genome contains any invalid data.
        bool validate() const;

        // Calculate and return distance between two genomes.
        static float calcDistance(const Genome& genome1, const Genome& genome2, const CalcDistParams& params);

    protected:
        Network::EdgeIds m_innovations;         // A list of innovations sorted by innovation id.
        InnovationCounter& m_innovIdCounter;    // The innovation counter shared by all the genomes.
    };
}
