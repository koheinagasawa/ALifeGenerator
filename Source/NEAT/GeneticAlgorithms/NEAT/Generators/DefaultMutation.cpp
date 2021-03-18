/*
* DefaultMutation.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Generators/DefaultMutation.h>
#include <NEAT/GeneticAlgorithms/Base/GenerationBase.h>
#include <NEAT/GeneticAlgorithms/Base/Selectors/GenomeSelector.h>

using namespace NEAT;

void DefaultMutation::mutate(GenomeBase* genomeInOut, MutationOut& mutationOut)
{
    mutationOut.clear();

    assert(m_params.m_weightMutationRate >= 0 && m_params.m_weightMutationRate <= 1);
    assert(m_params.m_weightMutationPerturbation >= 0 && m_params.m_weightMutationPerturbation <= 1);
    assert(m_params.m_weightMutationNewValRate >= 0 && m_params.m_weightMutationNewValRate <= 1);
    assert(m_params.m_weightMutationValMin <= m_params.m_weightMutationValMax);
    assert(m_params.m_addNodeMutationRate >= 0 && m_params.m_addNodeMutationRate <= 1);
    assert(m_params.m_addEdgeMutationRate >= 0 && m_params.m_addEdgeMutationRate <= 1);
    assert(m_params.m_newEdgeMinWeight <= m_params.m_newEdgeMaxWeight);

    Genome::NetworkPtr network = genomeInOut->accessNetwork();
    assert(network->validate());

    RandomGenerator* random = m_params.m_random ? m_params.m_random : &PseudoRandom::getInstance();

    int numNewEdges = 0;

    // 1. Change weights of edges with a certain probability
    for (const Genome::Network::EdgeEntry& edge : network->getEdges())
    {
        EdgeId edgeId = edge.first;
        if (random->randomReal01() <= m_params.m_weightMutationRate)
        {
            if (random->randomReal01() <= m_params.m_weightMutationNewValRate)
            {
                // Assign a completely new random weight.
                network->setWeight(edgeId, random->randomReal(m_params.m_weightMutationValMin, m_params.m_weightMutationValMax));
            }
            else
            {
                // Mutate the current weight by small perturbation.
                float weight = network->getWeight(edgeId);
                const float perturbation = random->randomReal(-m_params.m_weightMutationPerturbation, m_params.m_weightMutationPerturbation);
                weight = weight * (1.0f + perturbation);
                weight = std::max(m_params.m_weightMutationValMin, std::min(m_params.m_weightMutationValMax, weight));
                network->setWeight(edgeId, weight);
            }
        }
    }

    // 2. 3. Add a new node and edge

    // Decide whether we add a new node/edge
    const bool addNewNode = random->randomReal01() < m_params.m_addNodeMutationRate;
    const bool addNewEdge = random->randomReal01() < m_params.m_addEdgeMutationRate;

    // First, collect candidate edges/pairs of nodes where we can add new node/edge.
    // We do this now before we actually add any edge or node in order to prevent
    // mutation from happening more than once at the same element (e.g. adding a new edge at the newly added node).

    // Gather all edges which we can possibly add a new node.
    Genome::Network::EdgeIds edgeCandidates;
    if (addNewNode)
    {
        edgeCandidates.reserve(network->getNumEdges());
        for (const Genome::Network::EdgeEntry& edge : network->getEdges())
        {
            // We cannot add a new node at disable edges
            if (edge.second.isEnabled())
            {
                edgeCandidates.push_back(edge.first);
            }
        }
    }

    // Gather all pairs of nodes which we can possibly add a new edge.
    using NodePair = std::pair<NodeId, NodeId>;
    std::vector<NodePair> nodeCandidates;
    if (addNewEdge)
    {
        const Genome::Network::NodeDatas& nodeDatas = network->getNodes();
        nodeCandidates.reserve(nodeDatas.size() / 2);
        for (auto n1Itr = nodeDatas.cbegin(); n1Itr != nodeDatas.cend(); n1Itr++)
        {
            NodeId n1Id = n1Itr->first;
            const Genome::Node& n1 = network->getNode(n1Id);

            assert(n1.getNodeType() != Genome::Node::Type::NONE);

            auto n2Itr = n1Itr;
            n2Itr++;
            for (; n2Itr != nodeDatas.cend(); n2Itr++)
            {
                NodeId n2Id = n2Itr->first;
                const Genome::Node& n2 = network->getNode(n2Id);

                assert(n1.getNodeType() != Genome::Node::Type::NONE);

                // Cannot create an edge between two input nodes or two output nodes.
                if (n1.getNodeType() != Genome::Node::Type::HIDDEN && n1.getNodeType() == n2.getNodeType())
                {
                    continue;
                }

                // Check if these two nodes are already connected.
                if (network->isConnected(n1Id, n2Id) || network->isConnected(n2Id, n1Id))
                {
                    continue;
                }

                // Make sure that input node is not outNode and output node is not inNode.
                if (n1.getNodeType() == Genome::Node::Type::OUTPUT)
                {
                    std::swap(n1Id, n2Id);
                }
                else if (n2.getNodeType() == Genome::Node::Type::INPUT)
                {
                    std::swap(n1Id, n2Id);
                }

                nodeCandidates.push_back({ n1Id, n2Id });
            }
        }
    }

    // Function to assign innovation id to newly added edge and store its info in mutationOut.
    auto newEdgeAdded = [&mutationOut, network, &numNewEdges](EdgeId newEdge)
    {
        assert(numNewEdges < MutationOut::MAX_NUM_NEW_EDGES);

        // Store information of newly added edge.
        MutationOut::NewEdgeInfo& newEdgeInfo = mutationOut.m_newEdges[numNewEdges++];
        newEdgeInfo.m_sourceInNode = network->getInNode(newEdge);
        newEdgeInfo.m_sourceOutNode = network->getOutNode(newEdge);
        newEdgeInfo.m_newEdge = newEdge;
    };

    Genome* genome = static_cast<Genome*>(genomeInOut);

    // 2. Add a node at a random edge
    if (!edgeCandidates.empty())
    {
        // Select a random edge from candidates
        const EdgeId edgeToAddNode = edgeCandidates[random->randomInteger(0, (int)edgeCandidates.size() - 1)];
        NodeId newNode;
        EdgeId newIncomingEdge, newOutgoingEdge;
        genome->addNodeAt(edgeToAddNode, newNode, newIncomingEdge, newOutgoingEdge);

        newEdgeAdded(newIncomingEdge);
        newEdgeAdded(newOutgoingEdge);

        mutationOut.m_numNodesAdded++;
        mutationOut.m_newNode.m_newNode = newNode;
        mutationOut.m_newNode.m_previousEdgeId = edgeToAddNode;
        mutationOut.m_newNode.m_newIncomingEdgeId = newIncomingEdge;
        mutationOut.m_newNode.m_newOutgoingEdgeId = newOutgoingEdge;
        mutationOut.m_numEdgesAdded += 2;
    }

    assert(network->validate());

    // 3. Add an edge between random nodes
    if (!nodeCandidates.empty())
    {
        // Select a random node pair.
        const NodePair& pair = nodeCandidates[random->randomInteger(0, (int)nodeCandidates.size() - 1)];
        const float weight = random->randomReal(m_params.m_newEdgeMinWeight, m_params.m_newEdgeMaxWeight);
        bool tryAddFlippedEdgeOnFail = false;
        EdgeId newEdge = genome->addEdgeAt(pair.first, pair.second, weight, tryAddFlippedEdgeOnFail);

        if (!newEdge.isValid() &&
            network->getNode(pair.first).getNodeType() != Genome::Node::Type::INPUT &&
            network->getNode(pair.second).getNodeType() != Genome::Node::Type::OUTPUT)
        {
            // Adding edge was failed most likely because it will cause a circular network.
            // We might still be able to add an edge by flipping inNode and outNode when it's appropriate.
            newEdge = genome->addEdgeAt(pair.second, pair.first, weight, tryAddFlippedEdgeOnFail);
        }

        if (newEdge.isValid())
        {
            newEdgeAdded(newEdge);

            mutationOut.m_numEdgesAdded++;
        }
    }

    assert(network->validate());
}

void DefaultMutation::generate(int numTotalGenomes, int numRemaningGenomes, GenomeSelector* genomeSelector)
{
    assert(numTotalGenomes >= numRemaningGenomes);

    using GenomeData = GenerationBase::GenomeData;

    const int numGenomesToMutate = std::min(numRemaningGenomes, int(numTotalGenomes * m_params.m_mutatedGenomesRate));

    m_generatedGenomes.clear();;
    m_generatedGenomes.reserve(numGenomesToMutate);

    if (numGenomesToMutate <= 0)
    {
        return;
    }

    genomeSelector->preSelection(numGenomesToMutate, GenomeSelector::SELECT_ONE_GENOME);

    std::vector<MutationOut> mutationOuts;
    mutationOuts.resize(numGenomesToMutate);

    for (int i = 0; i < numGenomesToMutate; i++)
    {
        // Select a random genome.
        const GenomeData* gd = genomeSelector->selectGenome();

        // Copy genome in this generation first.
        GenomePtr newGenome = std::make_shared<Genome>(*std::static_pointer_cast<const Genome>(gd->getGenome()));

        // Mutate the genome.
        MutationOut& mout = mutationOuts[i];
        mutate(newGenome.get(), mout);

        // Check if there is already a mutation of the same structural change.
        // If so, assign the same innovation id to it.

        // Check all the newly added edges.
        for (int innov1 = 0; innov1 < mout.m_numEdgesAdded; innov1++)
        {
            MutationOut::NewEdgeInfo& newEdge = mout.m_newEdges[innov1];
            NodeId inNode = newEdge.m_sourceInNode;
            NodeId outNode = newEdge.m_sourceOutNode;

            bool idChanged = false;
            for (int j = 0; j < i; j++)
            {
                const MutationOut& mout2 = mutationOuts[j];
                for (int innov2 = 0; innov2 < mout2.m_numEdgesAdded; innov2++)
                {
                    const MutationOut::NewEdgeInfo& newEdge2 = mout2.m_newEdges[innov2];
                    if (newEdge2.m_sourceInNode == inNode && newEdge2.m_sourceOutNode == outNode)
                    {
                        newGenome->reassignInnovation(newEdge.m_newEdge, newEdge2.m_newEdge);
                        newEdge.m_newEdge = newEdge2.m_newEdge;
                        idChanged = true;
                        break;
                    }
                }

                if (idChanged)
                {
                    assert(newGenome->validate());
                    break;
                }
            }
        }


        // Check the newly added node.
        if (mout.m_numNodesAdded > 0)
        {
            MutationOut::NewNodeInfo& newNode = mout.m_newNode;
            EdgeId prevEdge = newNode.m_previousEdgeId;
            for (int j = 0; j < i; j++)
            {
                const MutationOut& mout2 = mutationOuts[j];
                const MutationOut::NewNodeInfo& newNode2 = mout2.m_newNode;

                if (mout2.m_numNodesAdded > 0 && mout2.m_newNode.m_previousEdgeId == prevEdge)
                {
                    newGenome->reassignNodeId(newNode.m_newNode, newNode2.m_newNode);
                    newGenome->reassignInnovation(newNode.m_newIncomingEdgeId, newNode2.m_newIncomingEdgeId);
                    newGenome->reassignInnovation(newNode.m_newOutgoingEdgeId, newNode2.m_newOutgoingEdgeId);
                    newNode.m_newNode = newNode2.m_newNode;
                    newNode.m_newIncomingEdgeId = newNode2.m_newIncomingEdgeId;
                    newNode.m_newOutgoingEdgeId = newNode2.m_newOutgoingEdgeId;

                    assert(newGenome->validate());
                    break;
                }
            }
        }

        m_generatedGenomes.push_back(std::static_pointer_cast<GenomeBase>(newGenome));
    }

    genomeSelector->postSelection();
}
