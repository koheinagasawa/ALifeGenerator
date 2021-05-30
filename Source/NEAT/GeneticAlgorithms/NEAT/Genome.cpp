/*
* Genome.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/NEAT/Genome.h>

#include <algorithm>

using namespace NEAT;

EdgeId InnovationCounter::getEdgeId(const EdgeEntry& entry)
{
    if (m_innovationHistory.find(entry) != m_innovationHistory.end())
    {
        return m_innovationHistory.at(entry);
    }

    EdgeId newEdge = m_innovationIdCounter.getNewId();
    m_innovationHistory[entry] = newEdge;
    return newEdge;
}

void InnovationCounter::reset()
{
    m_nodeIdCounter.reset();
    m_innovationIdCounter.reset();
    m_innovationHistory.clear();
}

Genome::Genome(const Cinfo& cinfo)
    : GenomeBase(cinfo.m_defaultActivation)
    , m_innovIdCounter(*cinfo.m_innovIdCounter)
{
    assert(cinfo.m_numInputNodes > 0 && cinfo.m_numOutputNodes > 0);

    Network::Nodes nodes;
    Network::Edges edges;
    Network::NodeIds outputNodes;
    Network::NodeIds inputNodes;

    const int numInputNodes = cinfo.m_numInputNodes + (cinfo.m_createBiasNode ? 1 : 0);
    const int numNodes = numInputNodes + cinfo.m_numOutputNodes;

    // Create nodes
    nodes.reserve(numNodes);
    inputNodes.reserve(cinfo.m_numInputNodes);
    outputNodes.reserve(cinfo.m_numOutputNodes);
    for (int i = 0; i < cinfo.m_numInputNodes; i++)
    {
        // Create input nodes.
        NodeId id = m_innovIdCounter.getNewNodeId();
        nodes.insert({ id, Node(Node::Type::INPUT) });
        inputNodes.push_back(id);
    }

    if (cinfo.m_createBiasNode)
    {
        // Create a bias node.
        m_biasNode = m_innovIdCounter.getNewNodeId();
        nodes.insert({ m_biasNode, Node(Node::Type::BIAS) });
        nodes[m_biasNode].setValue(cinfo.m_biasNodeValue);
        inputNodes.push_back(m_biasNode);
    }

    for (int i = numInputNodes; i < numNodes; i++)
    {
        // Create output nodes.
        NodeId id = m_innovIdCounter.getNewNodeId();
        nodes.insert({ id, Node(Node::Type::OUTPUT) });
        nodes[id].setActivation(m_defaultActivation);
        outputNodes.push_back(id);
    }

    // Create fully connected edges between input nodes and output nodes.
    // Input nodes are from 0 to numInputNodes and output nodes are from numInputNodes+1 to numNodes.
    const int numEdges = numInputNodes * cinfo.m_numOutputNodes;
    edges.reserve(numEdges);
    m_innovations.reserve(numEdges);
    {
        for (int i = 0; i < numInputNodes; i++)
        {
            for (int j = 0; j < cinfo.m_numOutputNodes; j++)
            {
                InnovationCounter::EdgeEntry entry{ inputNodes[i], outputNodes[j] };
                EdgeId eid = m_innovIdCounter.getEdgeId(entry);
                edges.insert({ eid, Network::Edge(inputNodes[i], outputNodes[j]) });
                m_innovations.push_back(eid);
            }
        }
    }

    // Create the network
    m_network = std::make_shared<Network>(nodes, edges, inputNodes, outputNodes);
}

Genome::Genome(const Genome& other)
    : GenomeBase(other)
    , m_innovations(other.m_innovations)
    , m_innovIdCounter(other.m_innovIdCounter)
{
}

void Genome::operator= (const Genome& other)
{
    assert(&m_innovIdCounter == &other.m_innovIdCounter);
    this->GenomeBase::operator=(other);
    m_innovations = other.m_innovations;
}

Genome::Genome(const Genome& source, NetworkPtr network, const Network::EdgeIds& innovations)
    : GenomeBase(source)
    , m_innovations(innovations)
    , m_innovIdCounter(source.m_innovIdCounter)
{
    m_network = network;

#ifdef _DEBUG
    {
        // Make sure that the network has the same number of input nodes as the source.
        int numInputNodes = 0;
        for (auto itr : network->getNodes())
        {
            if (itr.second.m_node.getNodeType() == Node::Type::INPUT)
            {
                numInputNodes++;
            }
        }
        assert(numInputNodes == (m_biasNode.isValid() ? ((int)source.getNetwork()->getInputNodes().size() - 1) : (int)source.getNetwork()->getInputNodes().size()));

        // Make sure that the number of innovations and the edges in the source are the same.
        assert(innovations.size() == network->getEdges().size());
    }
#endif
}

void Genome::addNodeAt(EdgeId edgeId, NodeId& newNode, EdgeId& newIncomingEdge, EdgeId& newOutgoingEdge)
{
    assert(m_network->hasEdge(edgeId));

    // Create new ids.
    newNode = m_innovIdCounter.getNewNodeId();
    NodeId inNode = m_network->getInNode(edgeId);
    NodeId outNode = m_network->getOutNode(edgeId);

    InnovationCounter::EdgeEntry entry1{ inNode, newNode };
    InnovationCounter::EdgeEntry entry2{ newNode, outNode };

    newIncomingEdge = m_innovIdCounter.getEdgeId(entry1);
    newOutgoingEdge = m_innovIdCounter.getEdgeId(entry2);

    // Add a node.
    bool result = m_network->addNodeAt(edgeId, newNode, newIncomingEdge, newOutgoingEdge);
    assert(result);

    // Set activation and mark it as a hidden node
    setNodeTypeAndActivation(newNode, Node::Type::HIDDEN, m_defaultActivation);

    // Record the innovations.
    m_innovations.push_back(newIncomingEdge);
    m_innovations.push_back(newOutgoingEdge);

    std::sort(m_innovations.begin(), m_innovations.end());
}

EdgeId Genome::addEdgeAt(NodeId inNode, NodeId outNode, float weight, bool tryAddFlippedEdgeOnFail)
{
    if (m_network->isConnected(inNode, outNode))
    {
        return EdgeId::invalid();
    }

    // Create a new id.
    InnovationCounter::EdgeEntry entry{ inNode, outNode };
    const EdgeId newEdge = m_innovIdCounter.getEdgeId(entry);

    // Add an edge.
    bool result = m_network->addEdgeAt(inNode, outNode, newEdge, weight);
    if (tryAddFlippedEdgeOnFail && !result)
    {
        result = m_network->addEdgeAt(outNode, inNode, newEdge, weight);
        assert(result);
    }

    if(result)
    {
        // Record the innovation.
        m_innovations.push_back(newEdge);
        std::sort(m_innovations.begin(), m_innovations.end());

        return newEdge;
    }
    else
    {
        return EdgeId::invalid();
    }
}

void Genome::removeEdge(EdgeId edge)
{
    assert(m_network->hasEdge(edge));

    // Remove the edge from the network.
    m_network->removeEdge(edge);
    
    // Remove the innovation.
    for (auto itr = m_innovations.begin(); itr != m_innovations.end(); itr++)
    {
        if (*itr == edge)
        {
            m_innovations.erase(itr);
            break;
        }
    }
}

void Genome::reassignNodeId(const NodeId originalId, const NodeId newId)
{
    assert(m_network->hasNode(originalId) && !m_network->hasNode(newId));

    const bool isInputNode = m_network->getNode(originalId).getNodeType() == Node::Type::INPUT;

    m_network->replaceNodeId(originalId, newId);

    assert(validate());
}

void Genome::reassignInnovation(const EdgeId originalId, const EdgeId newId)
{
    assert(m_network->hasEdge(originalId) && !m_network->hasEdge(newId));

    // Remove the original edge and add the new one.
    m_network->replaceEdgeId(originalId, newId);

    // Fix m_innovations. We perform reverse iteration here because this function is
    // typically called to fix newly added edges after mutation.
    // Add the new edge id to the innovation list
    for (auto itr = m_innovations.rbegin(); itr != m_innovations.rend(); itr++)
    {
        if (*itr < newId)
        {
            m_innovations.insert(itr.base(), newId);
            break;
        }
    }
    // Remove the original edge id from the innovation list
    for (auto itr = m_innovations.rbegin(); itr != m_innovations.rend(); itr++)
    {
        if (*itr == originalId)
        {
            m_innovations.erase((itr+1).base());
            break;
        }
    }

    assert(validate());
}

float Genome::calcDistance(const Genome& genome1, const Genome& genome2, const CalcDistParams& params)
{
    assert(genome1.validate());
    assert(genome2.validate());

    const Network* network1 = genome1.getNetwork();
    const Network* network2 = genome2.getNetwork();

    float disjointFactor = params.m_disjointFactor;
    // Normalize disjoint factor
    {
        const int numEdges1 = network1->getNumEdges();
        const int numEdges2 = network2->getNumEdges();
        const int numEdges = numEdges1 > numEdges2 ? numEdges1 : numEdges2;
        if (numEdges >= params.m_edgeNormalizationThreshold)
        {
            disjointFactor /= (float)numEdges;
        }
    }

    int numDisjointEdges = 0;
    int numMatchingEdges = 0;
    float sumWeightDiffs = 0.f;

    // Iterate over all edges in both genomes including disabled edges then
    // count the number of disjoint edges and calculate sum of weight differences.
    const Network::EdgeIds& innovations1 = genome1.getInnovations();
    const Network::EdgeIds& innovations2 = genome2.getInnovations();
    size_t curIdx1 = 0;
    size_t curIdx2 = 0;
    while (curIdx1 < innovations1.size() && curIdx2 < innovations2.size())
    {
        const EdgeId cur1 = innovations1[curIdx1];
        const EdgeId cur2 = innovations2[curIdx2];
        if (cur1 == cur2)
        {
            sumWeightDiffs += fabs(network1->getWeight(cur1) - network2->getWeight(cur2));
            curIdx1++;
            curIdx2++;
            numMatchingEdges++;
        }
        else
        {
            if (cur1 < cur2)
            {
                curIdx1++;
            }
            else
            {
                curIdx2++;
            }
            numDisjointEdges++;
        }
    }

    while (curIdx1++ < innovations1.size())
    {
        numDisjointEdges++;
    }
    while (curIdx2++ < innovations2.size())
    {
        numDisjointEdges++;
    }

    // Calculate the final distance
    return disjointFactor * numDisjointEdges + params.m_weightFactor * sumWeightDiffs / (float)numMatchingEdges;
}

bool Genome::validate() const
{
#ifdef DEBUG_SLOW
    // Make sure that the network is valid.
    if (!m_network.get()) return false;
    if (!m_network->validate()) return false;

    // Make sure that the number of innovations is correct.
    if (m_innovations.empty()) return false;
    if (m_innovations.size() != m_network->getNumEdges()) return false;

    // Make sure that the innovations are sorted
    EdgeId prev = m_innovations[0];
    if (!m_network->hasEdge(m_innovations[0])) return false;
    for (size_t i = 1; i < m_innovations.size(); i++)
    {
        const EdgeId& cur = m_innovations[i];

        if (!m_network->hasEdge(cur)) return false;
        if (prev >= cur) return false;

        prev = cur;
    }

    // Make sure that nodes are consistent
    {
        for (auto itr : m_network->getInputNodes())
        {
            if (m_network->getNode(itr).getNodeType() != Node::Type::INPUT && m_network->getNode(itr).getNodeType() != Node::Type::BIAS)
            {
                return false;
            }
        }

        for (auto itr : m_network->getOutputNodes())
        {
            if (m_network->getNode(itr).getNodeType() != Node::Type::OUTPUT)
            {
                return false;
            }
        }
    }
#endif
    return true;
}
