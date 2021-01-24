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
    : m_innovIdCounter(cinfo.m_innovIdCounter)
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
    PseudoRandom* random = params.m_random ? params.m_random : &PseudoRandom::getInstance();

    assert(params.m_weightMutationRate >= 0 && params.m_weightMutationRate <= 1);
    assert(params.m_weightMutationPerturbation >= 0 && params.m_weightMutationPerturbation <= 1);
    assert(params.m_weightMutationNewValRate >= 0 && params.m_weightMutationNewValRate <= 1);
    assert(params.m_weightMutationNewValMin <= params.m_weightMutationNewValMax);
    assert(params.m_addNodeMutationRate >= 0 && params.m_addNodeMutationRate <= 1);
    assert(params.m_addEdgeMutationRate >= 0 && params.m_addEdgeMutationRate <= 1);
    assert(params.m_newEdgeMinWeight <= params.m_newEdgeMaxWeight);

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
    auto newEdgeAdded = [&](EdgeId newEdge, NodeId inNode, NodeId outNode)
    {
        // Store this innovation
        InnovationEntry ie{ m_innovIdCounter.getNewInnovationId(), newEdge };
        m_innovations.push_back(ie);

        // Store information of newly added edge.
        MutationOut::NewEdgeInfo& newEdgeInfo = mutationOut.m_newEdges[numNewEdges++];
        newEdgeInfo.m_sourceInNode = inNode;
        newEdgeInfo.m_sourceOutNode = outNode;
        newEdgeInfo.m_newEdge = newEdge;
    };

    // 2. Add a node at a random edge
    if (random->randomReal01() < params.m_addNodeMutationRate)
    {
        // Randomly select an edge where we can add a node
        const int numEdges = m_network->getNumEdges();

        // Gather all edges which we can possibly add a new node.
        Network::EdgeIds edgeCandidates;
        edgeCandidates.reserve(numEdges);
        for (const Network::EdgeEntry& edge : m_network->getEdges())
        {
            // We cannot add a new node at disable edges
            if (edge.second.isEnabled())
            {
                edgeCandidates.push_back(edge.first);
            }
        }

        if (!edgeCandidates.empty())
        {
            // Select a random edge from candidates
            const EdgeId edgeToAddNode = edgeCandidates[random->randomInteger(0, (int)edgeCandidates.size() - 1)];

            // Add a new node and a new edge along with it.
            NodeId newNode;
            EdgeId newEdge;
            m_network->addNodeAt(edgeToAddNode, newNode, newEdge);

            assert(newNode.isValid());

            // Set it as a hidden node
            m_network->accessNode(newNode).m_type = Node::Type::HIDDEN;

            newEdgeAdded(newEdge, m_network->getInNode(edgeToAddNode), m_network->getOutNode(newEdge));
        }
    }

    // 3. Add an edge between random nodes
    if (random->randomReal01() < params.m_addEdgeMutationRate)
    {
        // Randomly select two nodes where we can add an edge
        const Network::NodeDatas& nodeDatas = m_network->getNodes();

        // Gather all pairs of nodes which we can possibly add a new edge.
        using NodePair = std::pair<NodeId, NodeId>;
        std::vector<NodePair> nodeCandidates;
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

                nodeCandidates.push_back({ n1Id, n2Id });
            }
        }

        if (!nodeCandidates.empty())
        {
            // Select a random node pair.
            const NodePair& pair = nodeCandidates[random->randomInteger(0, (int)nodeCandidates.size() - 1)];

            // Create a new edge.
            EdgeId newEdge = m_network->addEdgeAt(pair.first, pair.second, random->randomReal(params.m_newEdgeMinWeight, params.m_newEdgeMaxWeight));

            newEdgeAdded(newEdge, pair.first, pair.second);
        }
    }
}
