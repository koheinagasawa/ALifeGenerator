/*
* Genome.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/Genome.h>

#include <cassert>

using namespace NEAT;

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
    m_value = m_activation ? m_activation->activate(value) : value;
}

Genome::Genome(const Cinfo& cinfo)
    : m_innovIdCounter(*cinfo.m_innovIdCounter)
    , m_defaultActivation(cinfo.m_defaultActivation)
{
    assert(cinfo.m_numInputNodes > 0 && cinfo.m_numOutputNodes > 0);

    Network::Nodes nodes;
    Network::Edges edges;
    Network::NodeIds outputNodes;

    const int numNodes = cinfo.m_numInputNodes + cinfo.m_numOutputNodes;

    // Create nodes
    nodes.reserve(numNodes);
    m_inputNodes.resize(cinfo.m_numInputNodes);
    for (int i = 0; i < cinfo.m_numInputNodes; i++)
    {
        nodes.insert({ i, Node(Node::Type::INPUT) });
        m_inputNodes[i] = m_innovIdCounter.getNewNodeId();
    }
    for (int i = cinfo.m_numInputNodes; i < numNodes; i++)
    {
        nodes.insert({ i, Node(Node::Type::OUTPUT) });
        m_innovIdCounter.getNewNodeId();
    }

    // Create fully connected edges between input nodes and output nodes.
    // Input nodes are from 0 to numInputNodes and output nodes are from numInputNodes+1 to numNodes.
    const int numEdges = cinfo.m_numInputNodes * cinfo.m_numOutputNodes;
    edges.reserve(numEdges);
    m_innovations.reserve(numEdges);
    {
        for (int i = 0; i < cinfo.m_numInputNodes; i++)
        {
            for (int j = 0; j < cinfo.m_numOutputNodes; j++)
            {
                EdgeId eid = m_innovIdCounter.getNewInnovationId();
                edges.insert({ eid, Network::Edge(NodeId(i), NodeId(cinfo.m_numInputNodes + j)) });
                m_innovations.push_back(eid);
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

    // Set activation of output nodes
    if (m_defaultActivation)
    {
        for (NodeId nodeId : m_network->getOutputNodes())
        {
            m_network->accessNode(nodeId).setActivation(m_defaultActivation);
        }
    }
}

Genome::Genome(const Genome& other)
    : m_inputNodes(other.m_inputNodes)
    , m_innovations(other.m_innovations)
    , m_innovIdCounter(other.m_innovIdCounter)
    , m_defaultActivation(other.m_defaultActivation)
{
    m_network = std::make_shared<Network>(*other.m_network.get());
}

void Genome::operator= (const Genome& other)
{
    assert(&m_innovIdCounter == &other.m_innovIdCounter);
    m_inputNodes = other.m_inputNodes;
    m_innovations = other.m_innovations;
    m_defaultActivation = other.m_defaultActivation;
    m_network = std::make_shared<Network>(*other.m_network.get());
}

Genome::Genome(const Network::NodeIds& inputNodes, InnovationCounter& innovationCounter)
    : m_network(nullptr)
    , m_inputNodes(inputNodes)
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

    m_numNodesAdded = 0;
    m_numEdgesAdded = 0;

    m_newNode = NodeId::invalid();
}

void Genome::setActivationAll(const Activation* activation)
{
    assert(m_network.get());

    // Set activation for all hidden and output nodes.
    for (auto itr : m_network->getNodes())
    {
        Node& node = m_network->accessNode(itr.first);
        const Node::Type nodeType = node.getNodeType();
        if (nodeType == Node::Type::HIDDEN || nodeType == Node::Type::OUTPUT)
        {
            node.setActivation(activation);
        }
    }
}

void Genome::mutate(const MutationParams& params, MutationOut& mutationOut)
{
    assert(params.m_weightMutationRate >= 0 && params.m_weightMutationRate <= 1);
    assert(params.m_weightMutationPerturbation >= 0 && params.m_weightMutationPerturbation <= 1);
    assert(params.m_weightMutationNewValRate >= 0 && params.m_weightMutationNewValRate <= 1);
    assert(params.m_weightMutationValMin <= params.m_weightMutationValMax);
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
                m_network->setWeight(edgeId, random->randomReal(params.m_weightMutationValMin, params.m_weightMutationValMax));
            }
            else
            {
                // Mutate the current weight by small perturbation.
                float weight = m_network->getWeight(edgeId);
                const float perturbation = random->randomReal(-params.m_weightMutationPerturbation, params.m_weightMutationPerturbation);
                weight = weight * (1.0f + perturbation);
                weight = std::max(params.m_weightMutationValMin, std::min(params.m_weightMutationValMax, weight));
                m_network->setWeight(edgeId, weight);
            }
        }
    }

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

    // Function to assign innovation id to newly added edge and store its info in mutationOut.
    auto newEdgeAdded = [&](EdgeId newEdge)
    {
        assert(numNewEdges < MutationOut::NUM_NEW_EDGES);

        // Store this innovation
        m_innovations.push_back(newEdge);

        // Store information of newly added edge.
        MutationOut::NewEdgeInfo& newEdgeInfo = mutationOut.m_newEdges[numNewEdges++];
        newEdgeInfo.m_sourceInNode = m_network->getInNode(newEdge);
        newEdgeInfo.m_sourceOutNode = m_network->getOutNode(newEdge);
        newEdgeInfo.m_newEdge = newEdge;
    };

    // 2. Add a node at a random edge
    if (!edgeCandidates.empty())
    {
        // Select a random edge from candidates
        const EdgeId edgeToAddNode = edgeCandidates[random->randomInteger(0, (int)edgeCandidates.size() - 1)];

        // Add a new node and a new edge along with it.
        const NodeId newNode = m_innovIdCounter.getNewNodeId();
        const EdgeId newIncomingEdge = m_innovIdCounter.getNewInnovationId();
        const EdgeId newOutgoingEdge = m_innovIdCounter.getNewInnovationId();
        bool result = m_network->addNodeAt(edgeToAddNode, newNode, newIncomingEdge, newOutgoingEdge);
        assert(result);

        // Set activation and mark it as a hidden node
        {
            Node& nn = m_network->accessNode(newNode);
            nn.m_type = Node::Type::HIDDEN;
            nn.setActivation(m_defaultActivation);
        }


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

        assert(!m_network->isConnected(pair.first, pair.second));

        // Create a new edge.
        const float weight = random->randomReal(params.m_newEdgeMinWeight, params.m_newEdgeMaxWeight);
        const EdgeId newEdge = m_innovIdCounter.getNewInnovationId();
        bool result = m_network->addEdgeAt(pair.first, pair.second, newEdge, weight);
        if (!result)
        {
            // Invalid edge id was returned. This means adding this edge makes the network circular.
            // We should be able to add an edge of the opposite direction.
            result = m_network->addEdgeAt(pair.second, pair.first, newEdge, weight);
            assert(result);
        }

        newEdgeAdded(newEdge);

        mutationOut.m_numEdgesAdded++;
    }

    assert(m_network->validate());
}

Genome Genome::crossOver(const Genome& genome1, const Genome& genome2, bool sameFittingScore, const CrossOverParams& params)
{
    assert(genome1.validate());
    assert(genome2.validate());
    assert(&genome1.m_innovIdCounter == &genome2.m_innovIdCounter); // Make sure that the two genomes share the same innovation id counter.
    assert(genome1.getNetwork() && genome2.getNetwork());
    // Make sure that the numbers of input/output nodes are the same.
    // NOTE: Not only the number of nodes but also all node ids have to be identical too. Maybe we should check that here on debug.
    assert(genome1.m_inputNodes.size() == genome2.m_inputNodes.size());
    assert(genome1.getNetwork()->getNumOutputNodes() == genome2.getNetwork()->getNumOutputNodes());
    
    RandomGenerator& random = params.m_random ? *params.m_random : PseudoRandom::getInstance();

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
                if (random.randomReal01() < params.m_disablingEdgeRate)
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
                if (random.randomReal01() < params.m_matchingEdgeSelectionRate)
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
            sumWeightDiffs += abs(network1->getWeightRaw(cur1) - network2->getWeightRaw(cur2));
            curIdx1++;
            curIdx2++;
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
    return disjointFactor * numDisjointEdges + params.m_weightFactor * sumWeightDiffs;
}

void Genome::setInputNodeValues(const std::vector<float>& values) const
{
    assert(values.size() == m_inputNodes.size());

    for (int i = 0; i < (int)values.size(); i++)
    {
        m_network->setNodeValue(m_inputNodes[i], values[i]);
    }
}

void Genome::evaluate(const std::vector<float>& inputNodeValues) const
{
    setInputNodeValues(inputNodeValues);
    evaluate();
}

void Genome::evaluate() const
{
    assert(m_network.get());
    m_network->evaluate();
}

bool Genome::validate() const
{
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

    // Make sure that input nodes are inconsistent
    {
        int numInputNodes = 0;
        for (auto itr : m_network->getNodes())
        {
            if (m_network->getNode(itr.first).getNodeType() == Node::Type::INPUT)
            {
                numInputNodes++;
            }
        }

        if (numInputNodes != (int)m_inputNodes.size()) return false;

        for (NodeId nodeId : m_inputNodes)
        {
            if (!m_network->hasNode(nodeId)) return false;
            if (m_network->getNode(nodeId).getNodeType() != Node::Type::INPUT) return false;
        }
    }

    return true;
}
