/*
* DefaultCrossOver.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Generators/DefaultCrossOver.h>
#include <NEAT/GeneticAlgorithms/NEAT/Selectors/SpeciesBasedGenomeSelector.h>
#include <NEAT/GeneticAlgorithms/Base/Selectors/GenomeSelector.h>
#include <NEAT/NeuralNetwork/NeuralNetworkFactory.h>

using namespace NEAT;

auto DefaultCrossOver::crossOver(const GenomeBase& genome1In, const GenomeBase& genome2In, bool sameFittingScore)->GenomeBasePtr
{
    using Network = Genome::Network;

    const Genome& genome1 = *static_cast<const Genome*>(&genome1In);
    const Genome& genome2 = *static_cast<const Genome*>(&genome2In);

    assert(genome1.validate());
    assert(genome2.validate());
    assert(genome1.getNetwork() && genome2.getNetwork());
    // Make sure that the numbers of input/output nodes are the same.
    // NOTE: Not only the number of nodes but also all node ids have to be identical too. Maybe we should check that here on debug.
    assert(genome1.getNetwork()->getInputNodes().size() == genome2.getNetwork()->getInputNodes().size());
    assert(genome1.getNetwork()->getOutputNodes().size() == genome2.getNetwork()->getOutputNodes().size());

    RandomGenerator& random = m_params.m_random ? *m_params.m_random : PseudoRandom::getInstance();

    const Network* network1 = genome1.getNetwork();
    const Network* network2 = genome2.getNetwork();

    const Network::EdgeIds& innovations1 = genome1.getInnovations();
    const Network::EdgeIds& innovations2 = genome2.getInnovations();

    const bool allowCircularNetwork = network1->allowsCircularNetwork();

    // Create arrays to store innovations, nodes and edges for the new genome.
    Network::EdgeIds innovations;
    Network::Nodes newGenomeNodes;
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
        auto addEdge = [&](const EdgeId edgeId, const Genome* genomeA, const Genome* genomeB, bool sameFitnessDisjoint)
        {
            // Copy the edge
            const Genome::Edge& edgeA = genomeA->getNetwork()->getEdge(edgeId);
            Genome::Edge edge = edgeA;
            edge.setEnabled(true);

            // Disable the edge at a certain probability if either parent's edge is already disable
            if (!edgeA.isEnabled() || (genomeB && !genomeB->isEdgeEnabled(edgeId)))
            {
                if (random.randomReal01() < m_params.m_disablingEdgeRate)
                {
                    edge.setEnabled(false);
                }
                else
                {
                    if (!allowCircularNetwork && !sameFitnessDisjoint)
                    {
                        // Remember the edge which might be disabled in the parents but is enabled now.
                        // When fitness of the parents are the same (sameFitnessDisjoint == true), we don't need to
                        // store the edge because we will store it in disjointEnableEdges array below.
                        enabledEdges.push_back(edgeId);
                    }
                }
            }

            if (!allowCircularNetwork && sameFitnessDisjoint && edge.isEnabled())
            {
                disjointEnableEdges.push_back(edgeId);
            }

            newGenomeEdges.insert({ edgeId,  edge });
            assert(innovations.empty() || edgeId > innovations.back());
            innovations.push_back(edgeId);
        };

        // Iterate over all edges in both genomes including disabled edges.
        size_t curIdx1 = 0;
        size_t curIdx2 = 0;
        while (curIdx1 < innovations1.size() && curIdx2 < innovations2.size())
        {
            const EdgeId cur1 = innovations1[curIdx1];
            const EdgeId cur2 = innovations2[curIdx2];

            if (cur1 == cur2)
            {
                assert(network1->getInNode(cur1) == network2->getInNode(cur2));
                assert(network1->getOutNode(cur1) == network2->getOutNode(cur2));
                const bool isDisjoint = false;

                // Randomly select an edge from either genome1 or genome2 for matching edges.
                if (random.randomReal01() < m_params.m_matchingEdgeSelectionRate)
                {
                    addEdge(cur1, &genome1, &genome2, isDisjoint);
                }
                else
                {
                    addEdge(cur2, &genome2, &genome1, isDisjoint);
                }
                curIdx1++;
                curIdx2++;
            }
            else if (cur1 < cur2)
            {
                // Always take disjoint edges from more fit genome.
                addEdge(cur1, &genome1, nullptr, sameFittingScore);
                curIdx1++;
            }
            else
            {
                // Don't take disjoint edges from less fit genome unless the two genomes have the same fitness.
                if (sameFittingScore)
                {
                    addEdge(cur2, &genome2, nullptr, sameFittingScore);
                }
                curIdx2++;
            }
        }

        // Add all remaining excess edges
        if (!sameFittingScore)
        {
            // Only take edges of the more fit genome.
            const bool isSameFitnessDisjoint = false;
            while (curIdx1 < innovations1.size())
            {
                addEdge(innovations1[curIdx1++], &genome1, nullptr, isSameFitnessDisjoint);
            }
        }
        else
        {
            // Take edges from both genomes when they have the same fitness.
            const bool isSameFitnessDisjoint = true;
            while (curIdx1 < innovations1.size())
            {
                addEdge(innovations1[curIdx1++], &genome1, nullptr, isSameFitnessDisjoint);
            }
            while (curIdx2 < innovations2.size())
            {
                addEdge(innovations2[curIdx2++], &genome2, nullptr, isSameFitnessDisjoint);
            }
        }
    }

    // Add all nodes which are connected edges we've added above.
    {
        std::unordered_set<NodeId> addedNodes;
        for (auto& itr : newGenomeEdges)
        {
            const Genome::Edge& edge = itr.second;
            const NodeId inNode = edge.getInNode();
            const NodeId outNode = edge.getOutNode();

            // Add in node.
            if (addedNodes.find(inNode) == addedNodes.end())
            {
                newGenomeNodes.insert({ inNode, network1->hasNode(inNode) ? network1->getNode(inNode) : network2->getNode(inNode) });
                addedNodes.insert(inNode);
            }

            // Add out node.
            if (addedNodes.find(outNode) == addedNodes.end())
            {
                newGenomeNodes.insert({ outNode, network1->hasNode(outNode) ? network1->getNode(outNode) : network2->getNode(outNode) });
                addedNodes.insert(outNode);
            }
        }
    }

    // Add input, output and bias nodes in case we are missing any of them.
    {
        for (NodeId node : genome1.getNetwork()->getInputNodes())
        {
            newGenomeNodes.insert({ node, network1->getNode(node) });
        }
        for (NodeId node : network1->getOutputNodes())
        {
            newGenomeNodes.insert({ node, network1->getNode(node) });
        }

        NodeId biasNode = genome1.getBiasNode();
        if(biasNode.isValid())
        {
            newGenomeNodes.insert({ biasNode, network1->getNode(biasNode) });
        }
    }

    // Create a new network.
    Genome::NetworkPtr network = NeuralNetworkFactory::createNeuralNetwork<Genome::Node, SwitchableEdge>(
        network1->getType(),
        newGenomeNodes, newGenomeEdges,
        genome1.getNetwork()->getInputNodes(),
        genome1.getNetwork()->getOutputNodes());

    // In case of feed forward network, the child genome might have circular connections because some edges were enabled or due to disjoint edges inherited from the less fit genome.
    // Disable those edges one by one until we have no circular connection.
    if (!network->allowsCircularNetwork())
    {
        while (network->hasCircularEdges())
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

            assert(network->getEdge(edge).isEnabled());
            network->accessEdge(edge).setEnabled(false);
        }
    }

    assert(network->validate());

    // Create a new genome.
    return std::make_unique<Genome>(genome1, network, innovations);
}

void DefaultCrossOver::generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelector* genomeSelector)
{
    using GenomeData = GenerationBase::GenomeData;

    const int numGenomesToCrossover = std::min(numRemaningGenomes, (int)(numTotalGenomes * m_params.m_numCrossOverGenomesRate));

    if (numGenomesToCrossover <= 0)
    {
        // Nothing to select.
        return;
    }

    if(!genomeSelector->preSelection(numGenomesToCrossover, GenomeSelector::SELECT_TWO_GENOMES))
    {
        // Setting up genome selector was failed. Abort.
        return;
    }

    // Clear new genomes output.
    m_generatedGenomes.clear();
    m_generatedGenomes.reserve(numGenomesToCrossover);

    // Perform cross-over.
    for (int i = 0; i < numGenomesToCrossover; i++)
    {
        const GenomeData* g1 = nullptr;
        const GenomeData* g2 = nullptr;

        bool isSameFitness = false;

        // Select two genomes.
        {
            genomeSelector->selectTwoGenomes(g1, g2);
            assert(g1 && g2);

            // Swap g1 and g2 so that g1 has higher fitness
            const float fitness1 = g1->getFitness();
            const float fitness2 = g2->getFitness();
            if (fitness1 < fitness2)
            {
                std::swap(g1, g2);
            }
            else if (fitness1 == fitness2)
            {
                isSameFitness = true;
            }
        }

        // Cross-over.
        GenomeBasePtr newGenome = crossOver(*g1->getGenome(), *g2->getGenome(), isSameFitness);

        m_generatedGenomes.push_back(std::static_pointer_cast<GenomeBase>(newGenome));
    }

    genomeSelector->postSelection();
}
