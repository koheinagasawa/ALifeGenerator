/*
* Genome.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <memory>
#include <cstdint>

#include <NEAT/MutableNetwork.h>

DECLARE_ID(InnovationId, uint64_t);

// Singleton class to control InnovationId.
class InnovationCounter
{
public:
    InnovationCounter() = default;

    InnovationId getNewInnovationId();

    void reset();

protected:
    InnovationCounter(const InnovationCounter&) = delete;
    void operator=(const InnovationCounter&) = delete;

    InnovationId m_innovationCount = 0;
};

class Genome
{
public:
    struct InnovationEntry
    {
        InnovationId m_id;
        EdgeId m_edgeId;
    };

    struct Node : public NodeBase
    {
        virtual float getValue() const override { return m_value; }
        void setValue(float value) override { m_value = value; }
    protected:
        float m_value;
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

    // Get innovations of this network. Returned list of innovation entries is sorted by innovation id.
    inline auto getInnovations() const->const InnovationEntries& { return m_innovations; }

protected:
    NetworkPtr m_network;
    InnovationEntries m_innovations; // A list of innovations sorted by innovation id.
};
