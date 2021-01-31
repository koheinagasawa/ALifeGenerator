/*
* Genome.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/Genome.h>

#include <cassert>

using namespace NEAT;

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

Genome::Node::Node(Type type)
    : m_type(type)
{
}

float Genome::Node::getValue() const
{
    return m_value;
}

void Genome::Node::setValue(float value)
{
    assert(m_activation);
    m_value = m_activation->activate(value);
}

Genome::Genome(const Cinfo& cinfo)
    : m_innovIdCounter(*cinfo.m_innovIdCounter)
{
    assert(cinfo.m_numInputNodes > 0 && cinfo.m_numOutputNodes > 0);

    Network::Nodes nodes;
    Network::Edges edges;
    Network::NodeIds outputNodes;

    const int numNodes = cinfo.m_numInputNodes + cinfo.m_numOutputNodes;

    // Create nodes
    nodes.reserve(numNodes);
    for (int i = 0; i < cinfo.m_numInputNodes; i++)
    {
        nodes[i] = Node(Node::Type::INPUT);
    }
    for (int i = cinfo.m_numInputNodes; i < numNodes; i++)
    {
        nodes[i] = Node(Node::Type::OUTPUT);
    }

    // Create fully connected edges between input nodes and output nodes.
    // Input nodes are from 0 to numInputNodes and output nodes are from numInputNodes+1 to numNodes.
    const int numEdges = cinfo.m_numInputNodes * cinfo.m_numOutputNodes;
    edges.reserve(numEdges);
    m_innovations.reserve(numEdges);
    {
        EdgeId eid(0);
        for (int i = 0; i < cinfo.m_numInputNodes; i++)
        {
            for (int j = 0; j < cinfo.m_numOutputNodes; j++)
            {
                edges[eid] = Network::Edge(NodeId(i), NodeId(cinfo.m_numInputNodes + j));
                m_innovations.push_back({ m_innovIdCounter.getNewInnovationId(), eid });
                eid = eid.val() + 1;
            }
        }
    }

    // Store output node ids.
    outputNodes.reserve(cinfo.m_numOutputNodes);
    for (int i = 0; i < cinfo.m_numOutputNodes; i++)
    {
        outputNodes.push_back(NodeId(cinfo.m_numInputNodes + i));
    }

    // Create the network
    m_network = std::make_shared<Network>(nodes, edges, outputNodes);
}


Genome::Genome(InnovationCounter& innovationCounter)
    : m_network(nullptr)
    , m_innovIdCounter(innovationCounter)
{

}

void Genome::MutationOut::clear()
{
    for (int i = 0; i < NUM_NEW_EDGES; i++)
    {
        m_newEdges[i].m_sourceInNode = NodeId::invalid();
        m_newEdges[i].m_sourceOutNode = NodeId::invalid();
        m_newEdges[i].m_newEdge = EdgeId::invalid();
    }
}

void Genome::mutate(const MutationParams& params, MutationOut& mutationOut)
{
    assert(params.m_weightMutationRate >= 0 && params.m_weightMutationRate <= 1);
    assert(params.m_weightMutationPerturbation >= 0 && params.m_weightMutationPerturbation <= 1);
    assert(params.m_weightMutationNewValRate >= 0 && params.m_weightMutationNewValRate <= 1);
    assert(params.m_weightMutationNewValMin <= params.m_weightMutationNewValMax);
    assert(params.m_addNodeMutationRate >= 0 && params.m_addNodeMutationRate <= 1);
    assert(params.m_addEdgeMutationRate >= 0 && params.m_addEdgeMutationRate <= 1);
    assert(params.m_newEdgeMinWeight <= params.m_newEdgeMaxWeight);
    assert(m_network->validate());

    mutationOut.clear();

    RandomGenerator* random = params.m_random ? params.m_random : &PseudoRandom::getInstance();

    int numNewEdges = 0;

    // 1. Change weights of edges with a certain probability
    for (const Network::EdgeEntry& edge : m_network->getEdges())
    {
        EdgeId edgeId = edge.first;
        if (random->randomReal01() <= params.m_weightMutationRate)
        {
            if (random->randomReal01() <= params.m_weightMutationNewValRate)
            {
                // Assign a completely new random weight.
                m_network->setWeight(edgeId, random->randomReal(params.m_weightMutationNewValMin, params.m_weightMutationNewValMax));
            }
            else
            {
                // Mutate the current weight by small perturbation.
                float weight = m_network->getWeight(edgeId);
                const float perturbation = random->randomReal(-params.m_weightMutationPerturbation, params.m_weightMutationPerturbation);
                weight = weight * (1.0f + perturbation);
                m_network->setWeight(edgeId, weight);
            }
        }
    }

    // Function to assign innovation id to newly added edge and store its info in mutationOut.
    auto newEdgeAdded = [&](EdgeId newEdge)
    {
        // Store this innovation
        InnovationEntry ie{ m_innovIdCounter.getNewInnovationId(), newEdge };
        m_innovations.push_back(ie);

        // Store information of newly added edge.
        MutationOut::NewEdgeInfo& newEdgeInfo = mutationOut.m_newEdges[numNewEdges++];
        newEdgeInfo.m_sourceInNode = m_network->getInNode(newEdge);
        newEdgeInfo.m_sourceOutNode = m_network->getOutNode(newEdge);
        newEdgeInfo.m_newEdge = newEdge;
    };


    // 2. 3. Add a new node and edge

    // Decide whether we add a new node/edge
    const bool addNewNode = random->randomReal01() < params.m_addNodeMutationRate;
    const bool addNewEdge = random->randomReal01() < params.m_addEdgeMutationRate;

    // First, collect candidate edges/pairs of nodes where we can add new node/edge.
    // We do this now before we actually add any edge or node in order to prevent
    // mutation from happening more than once at the same element (e.g. adding a new edge at the newly added node).

    // Gather all edges which we can possibly add a new node.
    Network::EdgeIds edgeCandidates;
    if (addNewNode)
    {
        edgeCandidates.reserve(m_network->getNumEdges());
        for (const Network::EdgeEntry& edge : m_network->getEdges())
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
    if(addNewEdge)
    {
        const Network::NodeDatas& nodeDatas = m_network->getNodes();
        nodeCandidates.reserve(nodeDatas.size() / 2);
        for (auto n1Itr = nodeDatas.cbegin(); n1Itr != nodeDatas.cend(); n1Itr++)
        {
            NodeId n1Id = n1Itr->first;
            const Node& n1 = m_network->getNode(n1Id);

            assert(n1.m_type != Genome::Node::Type::NONE);

            auto n2Itr = n1Itr;
            n2Itr++;
            for (; n2Itr != nodeDatas.cend(); n2Itr++)
            {
                NodeId n2Id = n2Itr->first;
                const Node& n2 = m_network->getNode(n2Id);

                assert(n1.m_type != Genome::Node::Type::NONE);

                // Cannot create an edge between two input nodes or two output nodes.
                if (n1.m_type != Genome::Node::Type::HIDDEN && n1.m_type == n2.m_type)
                {
                    continue;
                }

                // Check if these two nodes are already connected.
                if (m_network->isConnected(n1Id, n2Id))
                {
                    continue;
                }

                // Make sure that input node is not outNode and output node is not inNode.
                if (n1.m_type == Genome::Node::Type::OUTPUT)
                {
                    std::swap(n1Id, n2Id);
                }
                else if (n2.m_type == Genome::Node::Type::INPUT)
                {
                    std::swap(n1Id, n2Id);
                }

                nodeCandidates.push_back({ n1Id, n2Id });
            }
        }
    }

    // 2. Add a node at a random edge
    if (!edgeCandidates.empty())
    {
        // Select a random edge from candidates
        const EdgeId edgeToAddNode = edgeCandidates[random->randomInteger(0, (int)edgeCandidates.size() - 1)];

        // Add a new node and a new edge along with it.
        NodeId newNode;
        EdgeId newIncomingEdge, newOutgoingEdge;
        m_network->addNodeAt(edgeToAddNode, newNode, newIncomingEdge, newOutgoingEdge);

        assert(newNode.isValid());

        // Set it as a hidden node
        m_network->accessNode(newNode).m_type = Node::Type::HIDDEN;

        newEdgeAdded(newIncomingEdge);
        newEdgeAdded(newOutgoingEdge);
    }

    // 3. Add an edge between random nodes
    if (!nodeCandidates.empty())
    {
        // Select a random node pair.
        const NodePair& pair = nodeCandidates[random->randomInteger(0, (int)nodeCandidates.size() - 1)];

        // Create a new edge.
        const float weight = random->randomReal(params.m_newEdgeMinWeight, params.m_newEdgeMaxWeight);
        EdgeId newEdge = m_network->addEdgeAt(pair.first, pair.second, weight);
        if (!newEdge.isValid())
        {
            // Invalid edge id was returned. This means adding this edge makes the network circular.
            // We should be able to add an edge of the opposite direction.
            newEdge = m_network->addEdgeAt(pair.second, pair.first, weight);
        }

        newEdgeAdded(newEdge);
    }

    assert(m_network->validate());
}

Genome Genome::crossOver(const Genome& genome1, const Genome& genome2, bool sameFittingScore, const CrossOverParams& params)
{
    assert(&genome1.m_innovIdCounter == &genome2.m_innovIdCounter); // Make sure that the two genomes share the same innovation id counter.
    assert(genome1.getNetwork() && genome2.getNetwork());
    assert(genome1.getNetwork()->getNumOutputNodes() == genome2.getNetwork()->getNumOutputNodes()); // Make sure that the numbers of output nodes are the same.

    RandomGenerator& random = params.m_random ? *params.m_random : PseudoRandom::getInstance();
    const Network* network1 = genome1.getNetwork();
    const Network* network2 = genome2.getNetwork();
    assert(network1->validate());
    assert(network2->validate());

    const InnovationEntries& innovations1 = genome1.getInnovations();
    const InnovationEntries& innovations2 = genome2.getInnovations();
    assert(innovations1.size() > 0);
    assert(innovations2.size() > 0);

#ifdef _DEBUG
    // Make sure that innovation entires are already sorted by innovation ids.
    {
        auto checkSorted = [](const InnovationEntries& entries)
        {
            InnovationId prevId = entries[0].m_id;
            for (size_t i = 1; i < entries.size(); i++)
            {
                const InnovationEntry& entry = entries[i];
                assert(prevId < entry.m_id);
                prevId = entry.m_id;
            }
        };

        checkSorted(innovations1);
        checkSorted(innovations2);
    }
#endif

    // Create a new genome and arrays to store nodes and edges for it.
    Genome newGenome(genome1.m_innovIdCounter);
    Network::Nodes nodes;
    Network::Edges edges;

    // Edges which are disabled in genome1 but enabled in the newGenome.
    // We need to keep track of them because they might make the network circular and might need to be disabled again.
    Network::EdgeIds enabledEdges;

    // Inherit edges
    {
        // Helper function to add an inherit edge.
        auto addEdge = [&](const InnovationEntry* entry1, const InnovationEntry* entry2 = nullptr, bool selectGenome1 = true)
        {
            // Copy the edge
            const Network::Edge& edge1 = network1->getEdges().at(entry1->m_edgeId);
            Network::Edge edge = edge1;
            edge.setEnabled(true);

            if (!selectGenome1)
            {
                edge.setWeight(network2->getWeight(entry2->m_edgeId));
            }

            // Disable the edge at a certain probability if either parent's edge is already disable
            bool disabled = !edge1.isEnabled() || (network2 && !network2->isEdgeEnabled(entry2->m_edgeId));
            if (disabled && random.randomReal01() < params.m_disablingEdgeRate)
            {
                edge.setEnabled(false);
            }
            else if (!edge1.isEnabled())
            {
                enabledEdges.push_back(entry1->m_edgeId);
            }

            edges[entry1->m_edgeId] = edge;
            newGenome.m_innovations.push_back(*entry1);
        };

        // Iterate over all edges in both genomes including disabled edges.
        size_t curIdx1 = 0;
        size_t curIdx2 = 0;
        while (curIdx1 < innovations1.size() && curIdx2 < innovations2.size())
        {
            const InnovationEntry& cur1 = innovations1[curIdx1];
            const InnovationEntry& cur2 = innovations2[curIdx2];

            if (cur1.m_id == cur2.m_id)
            {
                // Randomly select an edge from either genome1 or genome2 for matching edges.
                addEdge(&cur1, &cur2, random.randomReal01() < params.m_matchingEdgeSelectionRate);
                curIdx1++;
                curIdx2++;
            }
            else if (cur1.m_id < cur2.m_id)
            {
                // Always take disjoint edges from more fit genome.
                addEdge(&cur1);
                curIdx1++;
            }
            else
            {
                // Don't take disjoint edges from less fit genome.
                curIdx2++;
            }
        }

        // Add all remaining excess edges from genome1
        while(curIdx1 < innovations1.size())
        {
            addEdge(&innovations1[curIdx1++]);
        }
    }

    // Add all nodes which are connected edges we've added above.
    // [todo] We always inherit genome1's activation functions for all the nodes. Is there any way to select it based on fitness?
    {
        std::unordered_set<NodeId> addedNodes;
        for (auto& itr : edges)
        {
            const Network::Edge& edge = itr.second;
            NodeId inNode = edge.getInNode();
            NodeId outNode = edge.getOutNode();

            if (addedNodes.find(inNode) == addedNodes.end())
            {
                nodes[inNode] = network1->getNode(inNode);
                addedNodes.insert(inNode);
            }

            if (addedNodes.find(outNode) == addedNodes.end())
            {
                nodes[outNode] = network1->getNode(outNode);
                addedNodes.insert(outNode);
            }
        }
    }

    // Create a new network.
    newGenome.m_network = std::make_shared<Network>(nodes, edges, genome1.getNetwork()->getOutputNodes());

    // If the new network is not valid, it is likely that the network became circular because some edges were enabled.
    // Disable those edges one by one until we have a valid network.
    while (!newGenome.m_network->validate())
    {
        assert(enabledEdges.size() > 0);
        EdgeId edge = enabledEdges.back();
        enabledEdges.pop_back();
        assert(newGenome.m_network->isEdgeEnabled(edge));
        newGenome.m_network->setEdgeEnabled(edge, false);
    }

    return newGenome;
}
