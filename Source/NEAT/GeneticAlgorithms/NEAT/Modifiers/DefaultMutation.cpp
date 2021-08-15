/*
* DefaultMutation.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Modifiers/DefaultMutation.h>
#include <NEAT/GeneticAlgorithms/Base/GenerationBase.h>

using namespace NEAT;

void DefaultMutation::reset()
{
    m_mutations.clear();
}

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

    // 1. Change weights of edges with a small perturbation.
    for (const auto& edge : network->getEdges())
    {
        EdgeId edgeId = edge.getId();
        if (random->randomReal01() <= m_params.m_weightMutationRate)
        {
            if (random->randomReal01() <= m_params.m_weightMutationNewValRate)
            {
                // Assign a new random weight.
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

    Genome* genome = static_cast<Genome*>(genomeInOut);

    // 2. Change activation of a random node.
    NodeId nodeActivationMutated = NodeId::invalid();
    if (m_params.m_activationProvider && random->randomReal01() < m_params.m_changeActivationRate)
    {
        auto& nodes = network->accessNodes();

        // Select a random node.
        int index = random->randomInteger(0, nodes.size() - 1);
        Genome::Network::NodeData& nd = nodes[index];
        NodeId nodeId = nd.getId();

        if (nd.m_node.getNodeType() != DefaultNode::Type::BIAS && nd.m_node.getNodeType() != DefaultNode::Type::INPUT)
        {
            // Update activation function.
            const Activation* activation = m_params.m_activationProvider->getActivation();
            if (activation->m_id != nd.m_node.getActivationId())
            {
                nd.m_node.setActivation(activation);

                // Reset node id and ids of all the connected edges.
                genome->reassignNewNodeIdAndConnectedEdgeIds(nd.getId());
                nodeActivationMutated = nd.getId();
            }
        }
    }

    assert(network->validate());

    // 3. Remove a random existing edge.
    if (random->randomReal01() < m_params.m_removeEdgeMutationRate)
    {
        const auto& edges = network->getEdges();

        if (edges.size() > 1)
        {
            // Select a random edge to remove.
            int index = random->randomInteger(0, edges.size() - 1);

            // Find EdgeId of the edge.
            EdgeId edgeToRemove;
            {
                int i = 0;
                for (auto itr = edges.begin(); itr != edges.end(); itr++, i++)
                {
                    if (i == index)
                    {
                        edgeToRemove = itr->getId();
                        break;
                    }
                }
            }

            // Check if removing the edge will isolate an output node from the network. If so, don't remove it.
            NodeId outNodeId = network->getOutNode(edgeToRemove);
            if (network->getNode(outNodeId).getNodeType() != GenomeBase::Node::Type::OUTPUT ||
                network->getIncomingEdges(outNodeId).size() > 1)
            {
                // Remove the edge.
                genome->removeEdge(edgeToRemove);
            }
        }
    }

    assert(network->validate());

    // 4. 5. Add a new node and edge

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
        for (const auto& edgeData : network->getEdges())
        {
            const Genome::Edge& edge = edgeData.m_edge;
            NodeId inNode = edge.getInNode();
            NodeId outNode = edge.getOutNode();
            // We cannot add a new node at disable edges or edges from bias nodes.
            // We also skip node which mutated activation function in order to avoid applying multiple mutations to a single node at once.
            if (edge.isEnabled() && (network->getNode(inNode).getNodeType() != GenomeBase::Node::Type::BIAS) &&
                (!nodeActivationMutated.isValid() || (inNode != nodeActivationMutated && outNode != nodeActivationMutated)))
            {
                edgeCandidates.push_back(edgeData.getId());
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
            NodeId n1Id = n1Itr->getId();
            if (n1Id == nodeActivationMutated)
            {
                // Skip node which mutated activation function in order to avoid applying multiple mutations to a single node at once.
                continue;
            }

            const Genome::Node& n1 = network->getNode(n1Id);

            assert(n1.getNodeType() != Genome::Node::Type::NONE);

            auto n2Itr = n1Itr;
            n2Itr++;
            for (; n2Itr != nodeDatas.cend(); n2Itr++)
            {
                NodeId n2Id = n2Itr->getId();
                if (n2Id == nodeActivationMutated)
                {
                    // Skip node which mutated activation function in order to avoid applying multiple mutations to a single node at once.
                    continue;
                }

                const Genome::Node& n2 = network->getNode(n2Id);

                assert(n1.getNodeType() != Genome::Node::Type::NONE);

                // Cannot create an edge between two input nodes or two output nodes.
                if (n1.getNodeType() != Genome::Node::Type::HIDDEN && ((n1.getNodeType() == n2.getNodeType()) || (n1.isInputOrBias() && n2.isInputOrBias())))
                {
                    continue;
                }

                // Check if these two nodes are already connected.
                if (network->isConnected(n1Id, n2Id) || network->isConnected(n2Id, n1Id))
                {
                    continue;
                }

                // Make sure that input node is not outNode and output node is not inNode.
                if ((n1.getNodeType() == Genome::Node::Type::OUTPUT) || n2.isInputOrBias())
                {
                    nodeCandidates.push_back({ n2Id, n1Id });
                }
                else
                {
                    nodeCandidates.push_back({ n1Id, n2Id });
                }
            }
        }
    }

    // Function to assign innovation id to newly added edge and store its info in mutationOut.
    auto newEdgeAdded = [&mutationOut, network, &numNewEdges](EdgeId newEdge)
    {
        assert(numNewEdges < MutationOut::MAX_NUM_NEW_EDGES);

        // Store information of newly added edge.
        MutationOut::NewEdgeInfo& newEdgeInfo = mutationOut.m_newEdgeInfos[numNewEdges++];
        newEdgeInfo.m_sourceInNode = network->getInNode(newEdge);
        newEdgeInfo.m_sourceOutNode = network->getOutNode(newEdge);
        newEdgeInfo.m_edgeId = newEdge;
        mutationOut.m_numEdgesAdded++;
    };

    // 4. Add a node at a random edge
    if (!edgeCandidates.empty())
    {
        // Select a random edge from candidates
        const EdgeId edgeToAddNode = edgeCandidates[random->randomInteger(0, (int)edgeCandidates.size() - 1)];
        NodeId newNode;
        EdgeId newIncomingEdge, newOutgoingEdge;
        const Activation* activation = m_params.m_activationProvider ? m_params.m_activationProvider->getActivation() : nullptr;
        genome->addNodeAt(edgeToAddNode, activation, newNode, newIncomingEdge, newOutgoingEdge);

        newEdgeAdded(newIncomingEdge);
        newEdgeAdded(newOutgoingEdge);

        mutationOut.m_newNodeInfo.m_nodeId = newNode;
        mutationOut.m_newNodeInfo.m_activationId = activation ? activation->m_id : ActivationId::invalid();
        mutationOut.m_newNodeInfo.m_previousEdgeId = edgeToAddNode;
        mutationOut.m_newNodeInfo.m_newIncomingEdgeId = newIncomingEdge;
        mutationOut.m_newNodeInfo.m_newOutgoingEdgeId = newOutgoingEdge;
    }

    assert(network->validate());

    // 5. Add an edge between random nodes
    if (!nodeCandidates.empty())
    {
        // Select a random node pair.
        const NodePair& pair = nodeCandidates[random->randomInteger(0, (int)nodeCandidates.size() - 1)];
        const float weight = random->randomReal(m_params.m_newEdgeMinWeight, m_params.m_newEdgeMaxWeight);
        bool tryAddFlippedEdgeOnFail = false;
        EdgeId newEdge = genome->addEdgeAt(pair.first, pair.second, weight, tryAddFlippedEdgeOnFail);

        if (!newEdge.isValid() &&
            !network->getNode(pair.first).isInputOrBias() &&
            network->getNode(pair.second).getNodeType() != Genome::Node::Type::OUTPUT)
        {
            // Adding edge was failed most likely because it will cause a circular network.
            // We might still be able to add an edge by flipping inNode and outNode when it's appropriate.
            newEdge = genome->addEdgeAt(pair.second, pair.first, weight, tryAddFlippedEdgeOnFail);
        }

        if (newEdge.isValid())
        {
            newEdgeAdded(newEdge);
        }
    }

    assert(network->validate());
}

void DefaultMutation::modifyGenomes(GenomeBasePtr& genomeIn)
{
    if (!genomeIn)
    {
        return;
    }

    Genome* genome = static_cast<Genome*>(genomeIn.get());

    MutationOut mutationOut;
    mutate(genome, mutationOut);

    // Check if there is already a mutation of the same structural change.
    // If so, assign the same innovation id to it.
    // We iterate over the newly added nodes and check if there's any mutations with the same structural change.
    // Note that we don't need to check newly added edges between existing nodes because it is already guaranteed
    // that edges of the same structure get the same innovation id by NEAT::InnovationCounter.
    if (mutationOut.m_newNodeInfo.m_nodeId.isValid())
    {
        const MutationOut::NewNodeInfo& newNode = mutationOut.m_newNodeInfo;
        ActivationId activation = newNode.m_activationId;
        EdgeId prevEdge = newNode.m_previousEdgeId;
        int i = 0;
        for (; i < (int)m_mutations.size(); i++)
        {
            const MutationOut& mout2 = m_mutations[i];
            const MutationOut::NewNodeInfo& newNode2 = mout2.m_newNodeInfo;

            if (mout2.m_newNodeInfo.m_nodeId.isValid() && mout2.m_newNodeInfo.m_previousEdgeId == prevEdge && mout2.m_newNodeInfo.m_activationId == activation)
            {
                // The mutation is identical. Reassign IDs.
                genome->reassignNodeId(newNode.m_nodeId, newNode2.m_nodeId);
                genome->reassignInnovation(newNode.m_newIncomingEdgeId, newNode2.m_newIncomingEdgeId);
                genome->reassignInnovation(newNode.m_newOutgoingEdgeId, newNode2.m_newOutgoingEdgeId);

                assert(genome->validate());
                break;
            }
        }

        if (i == (int)m_mutations.size())
        {
            // Identical mutation wasn't found. Add the mutation to the list.
            m_mutations.push_back(mutationOut);
        }
    }
}
