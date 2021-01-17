/*
* Genome.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <cassert>

#include <NEAT/Genome.h>

InnovationId InnovationCounter::getNewInnovationId()
{
    InnovationId idOut = m_innovationCount;
    m_innovationCount = m_innovationCount.val() + 1;
    return idOut;
}

void InnovationCounter::reset()
{
    m_innovationCount = 0;
}

Genome::Genome(const Cinfo& cinfo)
{
    assert(cinfo.m_numInputNode > 0 && cinfo.m_numOutputNode > 0);

    Network::Nodes nodes;
    Network::Edges edges;
    Network::NodeIds outputNodes;

    const int numNodes = cinfo.m_numInputNode + cinfo.m_numOutputNode;

    nodes.reserve(numNodes);
    for (int i = 0; i < numNodes; i++)
    {
        nodes[i] = Node();
    }

    const int numEdges = cinfo.m_numInputNode * cinfo.m_numOutputNode;
    edges.reserve(numEdges);
    m_innovations.reserve(numEdges);
    {
        EdgeId eid(0);
        InnovationId iid(0);
        for (int i = 0; i < cinfo.m_numInputNode; i++)
        {
            for (int j = 0; j < cinfo.m_numOutputNode; j++)
            {
                edges[eid] = Network::Edge(NodeId(i), NodeId(cinfo.m_numInputNode + j));
                m_innovations.push_back({ iid, eid });
                eid = eid.val() + 1;
                iid = iid.val() + 1;
            }
        }
    }

    outputNodes.reserve(cinfo.m_numOutputNode);
    for (int i = 0; i < cinfo.m_numOutputNode; i++)
    {
        outputNodes.push_back(NodeId(cinfo.m_numInputNode + i));
    }

    m_network = std::make_shared<Network>(nodes, edges, outputNodes);
}
