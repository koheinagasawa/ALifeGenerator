/*
* DefaultMutation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/DefaultMutation.h>
#include <NEAT/Genome.h>

using namespace NEAT;

void DefaultMutation::mutate(GenomeBasePtr genomeIn, MutationOut& mutationOut)
{
    mutationOut.clear();

    assert(m_params.m_weightMutationRate >= 0 && m_params.m_weightMutationRate <= 1);
    assert(m_params.m_weightMutationPerturbation >= 0 && m_params.m_weightMutationPerturbation <= 1);
    assert(m_params.m_weightMutationNewValRate >= 0 && m_params.m_weightMutationNewValRate <= 1);
    assert(m_params.m_weightMutationValMin <= m_params.m_weightMutationValMax);
    assert(m_params.m_addNodeMutationRate >= 0 && m_params.m_addNodeMutationRate <= 1);
    assert(m_params.m_addEdgeMutationRate >= 0 && m_params.m_addEdgeMutationRate <= 1);
    assert(m_params.m_newEdgeMinWeight <= m_params.m_newEdgeMaxWeight);

    Genome::NetworkPtr network = genomeIn->accessNetwork();
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
                if (network->isConnected(n1Id, n2Id))
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
        assert(numNewEdges < MutationOut::NUM_NEW_EDGES);

        // Store information of newly added edge.
        MutationOut::NewEdgeInfo& newEdgeInfo = mutationOut.m_newEdges[numNewEdges++];
        newEdgeInfo.m_sourceInNode = network->getInNode(newEdge);
        newEdgeInfo.m_sourceOutNode = network->getOutNode(newEdge);
        newEdgeInfo.m_newEdge = newEdge;
    };

    Genome* genome = static_cast<Genome*>(genomeIn.get());

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
        mutationOut.m_newNode = newNode;
        mutationOut.m_numEdgesAdded += 2;
    }

    // 3. Add an edge between random nodes
    if (!nodeCandidates.empty())
    {
        // Select a random node pair.
        const NodePair& pair = nodeCandidates[random->randomInteger(0, (int)nodeCandidates.size() - 1)];
        const float weight = random->randomReal(m_params.m_newEdgeMinWeight, m_params.m_newEdgeMaxWeight);
        EdgeId newEdge = genome->addEdgeAt(pair.first, pair.second, weight);

        newEdgeAdded(newEdge);

        mutationOut.m_numEdgesAdded++;
    }

    assert(network->validate());
}

auto DefaultMutation::mutate(const GenomeDatas& generation, int numGenomesToMutate, GenomeSelectorBase* genomeSelector)->GenomeBasePtrs
{
    assert(genomeSelector);

    std::vector<MutationOut> mutationOuts;
    mutationOuts.resize(numGenomesToMutate);

    GenomeBasePtrs mutatedGenomesOut;
    mutatedGenomesOut.reserve(numGenomesToMutate);

    genomeSelector->setGenomes(generation);

    for (int i = 0; i < numGenomesToMutate; i++)
    {
        // Select a random genome.
        const GenomeData* gd = genomeSelector->selectGenome();

        //assert(gd->canReproduce());

        // Copy genome in this generation first.
        GenomePtr newGenome = std::make_shared<Genome>(*static_cast<const Genome*>(gd->getGenome()));

        // Mutate the genome.
        MutationOut& mout = mutationOuts[i];
        mutate(newGenome, mout);

        // Check if there is already a mutation of the same structural change.
        // If so, assign the same innovation id to it.
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
                    break;
                }
            }
        }

        mutatedGenomesOut.push_back(GenomeBasePtr(newGenome.get()));
    }

    return mutatedGenomesOut;
}
