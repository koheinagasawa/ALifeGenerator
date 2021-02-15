/*
* DefaultCrossOver.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/DefaultCrossOver.h>
#include <NEAT/Genome.h>

using namespace NEAT;

auto DefaultCrossOver::crossOver(const GenomeBase& genome1In, const GenomeBase& genome2In, bool sameFittingScore)->GenomeBase*
{
    using Network = Genome::Network;

    const Genome& genome1 = *static_cast<const Genome*>(&genome1In);
    const Genome& genome2 = *static_cast<const Genome*>(&genome2In);

    assert(genome1.validate());
    assert(genome2.validate());
    assert(genome1.getNetwork() && genome2.getNetwork());
    // Make sure that the numbers of input/output nodes are the same.
    // NOTE: Not only the number of nodes but also all node ids have to be identical too. Maybe we should check that here on debug.
    assert(genome1.getInputNodes().size() == genome2.getInputNodes().size());
    assert(genome1.getNetwork()->getNumOutputNodes() == genome2.getNetwork()->getNumOutputNodes());

    RandomGenerator& random = m_params.m_random ? *m_params.m_random : PseudoRandom::getInstance();

    const Network* network1 = genome1.getNetwork();
    const Network* network2 = genome2.getNetwork();

    const Network::EdgeIds& innovations1 = genome1.getInnovations();
    const Network::EdgeIds& innovations2 = genome2.getInnovations();

    // Create a new genome and arrays to store nodes and edges for it.
    Genome newGenome(genome1.m_inputNodes, genome1.m_innovIdCounter);
    Network::Nodes newGnomeNodes;
    Network::Edges newGenomeEdges;

    // Edges which are disabled in genome1 but enabled in the newGenome.
    // We need to keep track of them because they might make the network circular and might need to be disabled again.
    Network::EdgeIds enabledEdges;

    // List of disjoint edges. We populate this list only when sameFittingScore is true.
    // If fitnesses of the two genomes are the same, we are going to inherit structures from both genome 1 and genome2.
    // However, adding nodes/edges from both genomes could result a circular network. Then we remember disjoint edges first
    // and try to add them after we created a new genome by checking if adding such region won't invalidate the network.
    Network::EdgeIds disjointEnableEdges;

    // Inherit edges
    {
        // Helper function to add an inherit edge.
        auto addEdge = [&](const EdgeId edgeId, const Network* networkA, const Network* networkB, bool disjoint)
        {
            // Copy the edge
            const Network::Edge& edgeA = networkA->getEdges().at(edgeId);
            Network::Edge edge = edgeA;
            edge.setEnabled(true);

            // Disable the edge at a certain probability if either parent's edge is already disable
            bool disabled = !edgeA.isEnabled() || (networkB && !networkB->isEdgeEnabled(edgeId));
            if (disabled && !disjoint)
            {
                if (random.randomReal01() < m_params.m_disablingEdgeRate)
                {
                    edge.setEnabled(false);
                }
                else
                {
                    enabledEdges.push_back(edgeId);
                }
            }

            if (disjoint && edge.isEnabled())
            {
                disjointEnableEdges.push_back(edgeId);
            }

            newGenomeEdges.insert({ edgeId,  edge });
            assert(newGenome.m_innovations.empty() || edgeId > newGenome.m_innovations.back());
            newGenome.m_innovations.push_back(edgeId);
        };

        // Iterate over all edges in both genomes including disabled edges.
        size_t curIdx1 = 0;
        size_t curIdx2 = 0;
        while (curIdx1 < innovations1.size() && curIdx2 < innovations2.size())
        {
            const EdgeId cur1 = innovations1[curIdx1];
            const EdgeId cur2 = innovations2[curIdx2];
            bool isDisjoint;

            if (cur1 == cur2)
            {
                assert(network1->getInNode(cur1) == network2->getInNode(cur2));
                assert(network1->getOutNode(cur1) == network2->getOutNode(cur2));
                isDisjoint = false;

                // Randomly select an edge from either genome1 or genome2 for matching edges.
                if (random.randomReal01() < m_params.m_matchingEdgeSelectionRate)
                {
                    addEdge(cur1, network1, network2, isDisjoint);
                }
                else
                {
                    addEdge(cur2, network2, network1, isDisjoint);
                }
                curIdx1++;
                curIdx2++;
            }
            else if (cur1 < cur2)
            {
                // Always take disjoint edges from more fit genome.
                isDisjoint = sameFittingScore;
                addEdge(cur1, network1, nullptr, isDisjoint);
                curIdx1++;
            }
            else
            {
                // Don't take disjoint edges from less fit genome unless the two genomes have the same fitness.
                if (sameFittingScore)
                {
                    isDisjoint = true;
                    addEdge(cur2, network2, nullptr, isDisjoint);
                }
                curIdx2++;
            }
        }

        // Add all remaining excess edges
        if (!sameFittingScore)
        {
            const bool isDisjoint = false;
            while (curIdx1 < innovations1.size())
            {
                addEdge(innovations1[curIdx1++], network1, nullptr, isDisjoint);
            }
        }
        else
        {
            const bool isDisjoint = true;
            while (curIdx1 < innovations1.size())
            {
                addEdge(innovations1[curIdx1++], network1, nullptr, isDisjoint);
            }
            while (curIdx2 < innovations2.size())
            {
                addEdge(innovations2[curIdx2++], network2, nullptr, isDisjoint);
            }
        }
    }

    // Add all nodes which are connected edges we've added above.
    // [todo] We always inherit genome1's activation functions for all the nodes. Is there any way to select it based on fitness?
    {
        std::unordered_set<NodeId> addedNodes;
        for (auto& itr : newGenomeEdges)
        {
            const Network::Edge& edge = itr.second;
            const NodeId inNode = edge.getInNode();
            const NodeId outNode = edge.getOutNode();

            if (addedNodes.find(inNode) == addedNodes.end())
            {
                newGnomeNodes.insert({ inNode, network1->hasNode(inNode) ? network1->getNode(inNode) : network2->getNode(inNode) });
                addedNodes.insert(inNode);
            }

            if (addedNodes.find(outNode) == addedNodes.end())
            {
                newGnomeNodes.insert({ outNode, network1->hasNode(outNode) ? network1->getNode(outNode) : network2->getNode(outNode) });
                addedNodes.insert(outNode);
            }
        }
    }

    // Create a new network.
    newGenome.m_network = std::make_shared<Network>(newGnomeNodes, newGenomeEdges, genome1.getNetwork()->getOutputNodes());

    // If the new network is not valid, it is likely that the network became circular because some edges were enabled or due to disjoint edges.
    // Disable those edges one by one until we have a valid network.
    while (!newGenome.m_network->validate())
    {
        EdgeId edge;

        if (disjointEnableEdges.size() > 0)
        {
            // If there is any disjoint edges, try to disable them first.
            edge = disjointEnableEdges.back();
            disjointEnableEdges.pop_back();
        }
        else
        {
            // Then, try to disable newly enabled edges next.
            assert(enabledEdges.size() > 0);
            edge = enabledEdges.back();
            enabledEdges.pop_back();
        }

        assert(newGenome.m_network->isEdgeEnabled(edge));
        newGenome.m_network->setEdgeEnabled(edge, false);
    }

    return newGenome;
}
