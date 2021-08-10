/*
* TestUtils.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>
#include <UnitTest/Util/TestUtils.h>

// Helper function to compare two genomes' structure.
// Returns true if the two genomes have the same structure.
bool TestUtils::compareGenome(const NEAT::Genome& g1, const NEAT::Genome& g2)
{
    const NEAT::Genome::Network* net1 = g1.getNetwork();
    const NEAT::Genome::Network* net2 = g2.getNetwork();
    if (net1->getNumNodes() != net2->getNumNodes()) return false;
    if (net1->getNumEdges() != net2->getNumEdges()) return false;

    for (const auto& node : net1->getNodes())
    {
        NodeId id = node.getId();
        if (!net2->hasNode(id)) return false;
        const NEAT::Genome::Node& n1 = net1->getNode(id);
        const NEAT::Genome::Node& n2 = net2->getNode(id);

        if (n1.getNodeType() != n2.getNodeType()) return false;

        const auto& inEdges1 = net1->getIncomingEdges(id);
        const auto& inEdges2 = net2->getIncomingEdges(id);

        if (inEdges1.size() != inEdges2.size()) return false;

        for (int i = 0; i < (int)inEdges1.size(); i++)
        {
            if (inEdges1[i] != inEdges2[i]) return false;
        }
    }

    for (const auto& edge : net1->getEdges())
    {
        EdgeId id = edge.getId();
        if (!net2->hasEdge(id)) return false;

        const NEAT::Genome::Edge& e1 = edge.m_edge;
        const NEAT::Genome::Edge& e2 = net2->getEdge(id);

        if (e1.getInNode() != e2.getInNode()) return false;
        if (e1.getOutNode() != e2.getOutNode()) return false;
    }

    return true;
}

// Helper function to compare two genomes' structure and edge's weights and states.
// Returns true if the two genomes have the same structure, weights and states.
bool TestUtils::compareGenomeWithWeightsAndStates(const NEAT::Genome& g1, const NEAT::Genome& g2)
{
    if (!compareGenome(g1, g2)) return false;

    const NEAT::Genome::Network* net1 = g1.getNetwork();
    const NEAT::Genome::Network* net2 = g2.getNetwork();

    for (const auto& edge : net1->getEdges())
    {
        EdgeId id = edge.getId();
        if (!net2->hasEdge(id)) return false;

        const NEAT::Genome::Edge& e1 = edge.m_edge;
        const NEAT::Genome::Edge& e2 = net2->getEdge(id);

        if (e1.getWeightRaw() != e2.getWeightRaw()) return false;
        if (e1.isEnabled() != e2.isEnabled()) return false;
    }

    return true;
}
