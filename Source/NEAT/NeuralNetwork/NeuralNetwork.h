/*
* NeuralNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <cassert>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#include <Common/BaseType.h>
#include <NEAT/NeuralNetwork/Node.h>
#include <NEAT/NeuralNetwork/Edge.h>
#include <NEAT/NeuralNetwork/BakedNeuralNetwork.h>

// Type of neural network
enum class NeuralNetworkType
{
    GENERAL,
    FEED_FORWARD,
};

// Base class of neural network
template <typename Node, typename Edge>
class NeuralNetwork
{
public:
    //
    // Type Declarations
    //

    using NodeIds = std::vector<NodeId>;
    using EdgeIds = std::vector<EdgeId>;

    // Node and some additional data for shortcut access.
    struct NodeData
    {
    public:
        // Constructors
        NodeData() = default;
        NodeData(const Node& node, NodeId id);

        // Return incoming edges.
        inline auto getIncomingEdges() const->const EdgeIds& { return m_incomingEdges; }
        // Return outgoing edges.
        inline auto getOutgoingEdges() const->const EdgeIds& { return m_outgoingEdges; }
        // Get the id of this node.
        inline NodeId getId() const { return m_id; }

    public:
        Node m_node; // The node.

    private:
        EdgeIds m_incomingEdges;    // List of incoming edges to this node.
        EdgeIds m_outgoingEdges;    // List of outgoing edges from this node.
        NodeId m_id;                // Id of this node.

        // Intermediate data for evaluation
        enum class EvalState : char
        {
            NONE,
            EVALUATED
        };
        EvalState m_state;

        friend class NeuralNetwork;
    };

    using NodeDatas = std::unordered_map<NodeId, NodeData>;
    using Edges = std::unordered_map<EdgeId, Edge>;
    using Nodes = std::unordered_map<NodeId, Node>;

    //
    // Constructors
    //

    // Constructor from network information.
    NeuralNetwork(const Nodes& nodes, const Edges& edges);
    NeuralNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& inputNodes, const NodeIds& outputNodes);

    // Copy constructor
    NeuralNetwork(const NeuralNetwork& other) = default;

    // Create a copy of this network.
    virtual auto clone() const->std::shared_ptr<NeuralNetwork<Node, Edge>>;

    // Return true if this network can have circular edge connection.
    virtual bool allowsCircularNetwork() const { return true; }

    // Return type of this network.
    virtual NeuralNetworkType getType() const { return NeuralNetworkType::GENERAL; }

    //
    // Node Queries
    //

    // Return the number of nodes.
    inline int getNumNodes() const { return (int)m_nodes.size(); }

    // Return all the nodes in this network.
    inline auto getNodes() const->const NodeDatas& { return m_nodes; }
    inline auto accessNodes()->NodeDatas& { return m_nodes; }

    // Return true if the node id exists.
    inline bool hasNode(NodeId id) const;

    // Get read-only access to the node of given id.
    inline auto getNode(NodeId id) const->const Node&;

    // Get read-write access to the node of given id.
    inline auto accessNode(NodeId id)->Node&;

    // Get a list of incoming edges of the node.
    inline auto getIncomingEdges(NodeId id) const->const EdgeIds&;

    // Get a list of incoming edges of the node.
    inline auto getOutgoingEdges(NodeId id) const->const EdgeIds&;

    // Return true if the two nodes are connected.
    inline bool isConnected(NodeId node1, NodeId node2) const;

    // Set values of the all nodes.
    inline void setAllNodeValues(float value);

    // Set a value of the node.
    inline void setNodeValue(NodeId id, float value);

    // Get input node ids.
    inline auto getInputNodes() const->const NodeIds& { return m_inputNodes; }

    // Get output node ids.
    inline auto getOutputNodes() const->const NodeIds& { return m_outputNodes; }

    // Return true if this network has circular edge connections.
    bool hasCircularEdges() const;

    //
    // Edge Queries
    //

    // Return the number of edges.
    inline int getNumEdges() const { return (int)m_edges.size(); }

    // Return all the edges in the network.
    inline auto getEdges() const->const Edges& { return m_edges; }

    // Return true if the edge id exists.
    inline bool hasEdge(EdgeId id) const;

    // Return read-only access to an edge in the network.
    inline auto getEdge(EdgeId id) const->const Edge& { return m_edges.at(id); }

    // Return read-write access to an edge in the network.
    inline auto accessEdge(EdgeId id)->Edge& { return m_edges.at(id); }

    // Get the in-node of the edge.
    inline auto getInNode(EdgeId id) const->NodeId;

    // Get the out-node of the edge.
    inline auto getOutNode(EdgeId id) const->NodeId;

    // Get weight of the edge.
    inline float getWeight(EdgeId id) const;

    // Set weight of the edge.
    inline void setWeight(EdgeId id, float weight);

    //
    // Structural Modification
    //

    // Add a new node by dividing the edge at edgeId.
    bool addNodeAt(EdgeId edgeId, NodeId newNodeId, EdgeId newIncomingEdgeId, EdgeId newOutgoingEdgeId);

    // Add a new edge between node1 and node2 with weight.
    bool addEdgeAt(NodeId node1, NodeId node2, EdgeId newEdgeId, float weight = 1.0f);

    // Remove an existing edge
    void removeEdge(EdgeId edgeId);

    // Replace an node id with a new node id.
    void replaceNodeId(NodeId nodeId, NodeId newId);

    // Replace an edge id with a new edge id.
    void replaceEdgeId(EdgeId edgeId, EdgeId newId);

    // Return true if a new edge can be added between inNode and outNode.
    virtual bool canAddEdgeAt(NodeId inNode, NodeId outNode) const { return true; }

    //
    // Evaluation
    //

    // Evaluates this network and calculate new values for each node.
    virtual void evaluate();

    // Create a baked network.
    auto bake() const->std::shared_ptr<BakedNeuralNetwork>;

    //
    // Validation
    //

    // Return false if this network has invalid data.
    virtual bool validate() const;

protected:
    // Default constructor.
    NeuralNetwork() = default;

    // Construct m_nodes. This is called from constructor.
    void constructData(const Nodes& nodes, const Edges& edges);

    // Implementation of hasCircularEdges()
    bool hasCircularEdgesImpl(NodeId nodeId, std::unordered_set<NodeId>& checkedNodes) const;

    // Nodes of this network.
    NodeDatas m_nodes;

    // Edges of this network.
    Edges m_edges;

    NodeIds m_inputNodes; // A list of output nodes of this network.
    NodeIds m_outputNodes; // A list of output nodes of this network.
};

template <typename Node, typename Edge>
NeuralNetwork<Node, Edge>::NodeData::NodeData(const Node& node, NodeId id)
    : m_node(node)
    , m_id(id)
    , m_state(EvalState::NONE)
{
}

template <typename Node, typename Edge>
NeuralNetwork<Node, Edge>::NeuralNetwork(const Nodes& nodes, const Edges& edges)
    : m_edges(edges)
{
    constructData(nodes, edges);
}

template <typename Node, typename Edge>
NeuralNetwork<Node, Edge>::NeuralNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& inputNodes, const NodeIds& outputNodes)
    : m_edges(edges)
    , m_inputNodes(inputNodes)
    , m_outputNodes(outputNodes)
{
    constructData(nodes, edges);
}

template <typename Node, typename Edge>
auto NeuralNetwork<Node, Edge>::clone() const->std::shared_ptr<NeuralNetwork<Node, Edge>>
{
    return std::make_shared<NeuralNetwork>(*this);
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::constructData(const Nodes& nodes, const Edges& edges)
{
    // Allocate and add NodeDatas
    m_nodes.clear();
    m_nodes.reserve(nodes.size());
    for (const auto& itr : nodes)
    {
        m_nodes[itr.first] = NodeData(itr.second, itr.first);
    }

    // Set incoming edges and outgoing edges array in each node
    for (const auto& itr : edges)
    {
        const Edge& e = itr.second;
        NodeId outNode = e.getOutNode();
        if (!hasNode(outNode))
        {
            WARN("Input edge contains invalid outNode value.");
            continue;
        }
        m_nodes[outNode].m_incomingEdges.push_back(itr.first);

        NodeId inNode = e.getInNode();
        if (!hasNode(inNode))
        {
            WARN("Input edge contains invalid inNode value.");
            continue;
        }
        m_nodes[inNode].m_outgoingEdges.push_back(itr.first);
    }
}

template <typename Node, typename Edge>
inline bool NeuralNetwork<Node, Edge>::hasNode(NodeId id) const 
{
    return m_nodes.find(id) != m_nodes.end();
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getNode(NodeId id) const->const Node&
{
    assert(hasNode(id));
    return m_nodes.at(id).m_node;
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::accessNode(NodeId id)->Node&
{
    assert(hasNode(id));
    return m_nodes.at(id).m_node;
}

template <typename Node, typename Edge>
inline void NeuralNetwork<Node, Edge>::setAllNodeValues(float value)
{
    for (auto& elem : m_nodes)
    {
        elem.second.m_node.setValue(value);
    }
}

template <typename Node, typename Edge>
inline void NeuralNetwork<Node, Edge>::setNodeValue(NodeId id, float value)
{
    assert(hasNode(id));
    m_nodes[id].m_node.setValue(value);
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getIncomingEdges(NodeId id) const->const EdgeIds&
{
    return m_nodes.at(id).getIncomingEdges();
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getOutgoingEdges(NodeId id) const->const EdgeIds&
{
    return m_nodes.at(id).getOutgoingEdges();
}

template <typename Node, typename Edge>
inline bool NeuralNetwork<Node, Edge>::isConnected(NodeId node1, NodeId node2) const
{
    assert(hasNode(node1) && hasNode(node2) && node1 != node2);

    for (EdgeId e : getIncomingEdges(node1))
    {
        if (getInNode(e) == node2)
        {
            return true;
        }
    }

    for (EdgeId e : getOutgoingEdges(node1))
    {
        if (getOutNode(e) == node2)
        {
            return true;
        }
    }

    return false;
}

template <typename Node, typename Edge>
inline bool NeuralNetwork<Node, Edge>::hasEdge(EdgeId id) const
{
    return m_edges.find(id) != m_edges.end();
}

template <typename Node, typename Edge>
inline float NeuralNetwork<Node, Edge>::getWeight(EdgeId id) const
{
    assert(hasEdge(id));
    return m_edges.at(id).getWeight();
}

template <typename Node, typename Edge>
inline void NeuralNetwork<Node, Edge>::setWeight(EdgeId id, float weight)
{
    assert(hasEdge(id));
    m_edges[id].setWeight(weight);
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getInNode(EdgeId id) const->NodeId
{
    assert(hasEdge(id));
    return m_edges.at(id).getInNode();
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getOutNode(EdgeId id) const->NodeId
{
    assert(hasEdge(id));
    return m_edges.at(id).getOutNode();
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::hasCircularEdges() const
{
    const int numNodes = (int)m_nodes.size();
    std::unordered_set<NodeId> checkedNodes;

    // First, start looking at output nodes. This first path should cover most nodes already.
    for (NodeId id : m_outputNodes)
    {
        if (hasCircularEdgesImpl(id, checkedNodes))
        {
            return true;
        }

        checkedNodes.insert(id);
    }

    // Second, we iterate over all nodes while skipping nodes which are already checked in the previous loop.
    // This is needed to find any circles isolated from the network containing the output nodes.
    for (const auto& elem : m_nodes)
    {
        if (checkedNodes.find(elem.first) != checkedNodes.end())
        {
            continue;
        }

        if(hasCircularEdgesImpl(elem.first, checkedNodes))
        {
            return true;
        }

        checkedNodes.insert(elem.first);
    }

    return false;
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::hasCircularEdgesImpl(NodeId startNodeId, std::unordered_set<NodeId>& checkedNodes) const
{
    std::unordered_set<NodeId> visitingNodes;
    std::vector<NodeId> nodeStack;
    nodeStack.push_back(startNodeId);

    while (nodeStack.size() > 0)
    {
        NodeId currentNodeId = nodeStack.back();
        visitingNodes.insert(currentNodeId);

        bool newNodeInStack = false;

        // Follow edges backward and see if we visit the same node more than once.
        for (EdgeId e : getIncomingEdges(currentNodeId))
        {
            const Edge& edge = getEdge(e);

            // Ignore disabled edges.
            if (!edge.isEnabled())
            {
                continue;
            }

            NodeId inNodeId = edge.getInNode();
            if (visitingNodes.find(inNodeId) != visitingNodes.end())
            {
                // Found a circle
                return true;
            }

            // Skip this node if we have already looked at it.
            if (checkedNodes.find(inNodeId) != checkedNodes.end())
            {
                continue;
            }

            // Add inNodeId to the stack and continue to follow that path.
            nodeStack.push_back(inNodeId);
            newNodeInStack = true;
            break;
        }

        if (newNodeInStack)
        {
            continue;
        }

        // The current node is not a part of a circle.
        visitingNodes.erase(currentNodeId);
        checkedNodes.insert(currentNodeId);
        nodeStack.pop_back();
    }

    return false;
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::addNodeAt(EdgeId edgeId, NodeId newNodeId, EdgeId newIncomingEdgeId, EdgeId newOutgoingEdgeId)
{
    assert(validate());
    assert(!hasNode(newNodeId) && !hasEdge(newIncomingEdgeId) && !hasEdge(newOutgoingEdgeId));

    // Make sure that edgeId exists.
    if(!hasEdge(edgeId))
    {
        WARN("Edge id doesn't exist.");
        return false;
    }

    NodeId inNode, outNode;
    {
        const Edge& edgeToDivide = getEdge(edgeId);
        inNode = edgeToDivide.getInNode();
        outNode = edgeToDivide.getOutNode();
    }

    // Create two new edges.
    m_edges[newIncomingEdgeId] = Edge(inNode, newNodeId, 1.0f);
    m_edges[newOutgoingEdgeId] = Edge(newNodeId, outNode, 1.0f);

    // Create a new node.
    NodeData newNode;
    newNode.m_incomingEdges.push_back(newIncomingEdgeId);
    newNode.m_outgoingEdges.push_back(newOutgoingEdgeId);
    newNode.m_id = newNodeId;
    m_nodes[newNodeId] = newNode;

    // Update incoming edge and outgoing edge of the existing node.
    m_nodes[inNode].m_outgoingEdges.push_back(newIncomingEdgeId);
    m_nodes[outNode].m_incomingEdges.push_back(newOutgoingEdgeId);

    assert(validate());

    return true;
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::addEdgeAt(NodeId node1, NodeId node2, EdgeId newEdgeId, float weight /* = 1.0f */)
{
    assert(validate());
    assert(!hasEdge(newEdgeId));

    // Make sure that node1 and node2 exist.
    if (!hasNode(node1) || !hasNode(node2))
    {
        WARN("At least one of the give node ids doesn't exist.");
        return false;
    }

    EdgeIds& edgesFromInNode = m_nodes[node1].m_outgoingEdges;
    EdgeIds& edgesToOutNode = m_nodes[node2].m_incomingEdges;

    // Check if there is already an edge between the two nodes.
    for (EdgeId eid : edgesToOutNode)
    {
        if (getInNode(eid) == node1)
        {
            WARN("There is already an edge between the given two nodes.");
            return false;
        }
    }

    // Abort if we cannot add the edge here.
    if (!canAddEdgeAt(node1, node2))
    {
        return false;
    }

    // Create a new edge
    m_edges[newEdgeId] = Edge(node1, node2, weight);
    edgesFromInNode.push_back(newEdgeId);
    edgesToOutNode.push_back(newEdgeId);

    assert(validate());

    return true;
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::replaceNodeId(NodeId nodeId, NodeId newId)
{
    assert(validate());
    assert(hasNode(nodeId));
    assert(!hasNode(newId));

    // Replace nodeIds stored in edges.
    for (auto& elem : m_edges)
    {
        const Edge& curEdge = elem.second;
        if (curEdge.getInNode() == nodeId)
        {
            Edge newEdge(newId, curEdge.getOutNode());
            newEdge.copyState(&curEdge);
            elem.second = newEdge;
        }
        else if (curEdge.getOutNode() == nodeId)
        {
            Edge newEdge(curEdge.getInNode(), newId);
            newEdge.copyState(&curEdge);
            elem.second = newEdge;
        }
    }

    // Replace the node itself
    {
        NodeData nd = m_nodes.at(nodeId);
        nd.m_id = newId;
        m_nodes.erase(nodeId);
        m_nodes[newId] = nd;
    }

    // Update lists of input/output nodes
    {
        bool isInput = false;

        for (auto itr = m_inputNodes.begin(); itr != m_inputNodes.end(); itr++)
        {
            if (*itr == nodeId)
            {
                m_inputNodes.erase(itr);
                m_inputNodes.push_back(newId);
                isInput = true;
                break;
            }
        }

        if (!isInput)
        {
            for (auto itr = m_outputNodes.begin(); itr != m_outputNodes.end(); itr++)
            {
                if (*itr == nodeId)
                {
                    m_outputNodes.erase(itr);
                    m_outputNodes.push_back(newId);
                    break;
                }
            }
        }
    }

    assert(validate());
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::removeEdge(EdgeId edgeId)
{
    assert(validate());
    assert(hasEdge(edgeId));

    const Edge& edge = m_edges.at(edgeId);

    // Update incoming edges and outgoing edges of the in/out nodes
    {
        EdgeIds& edgesFromInNode = m_nodes[edge.getInNode()].m_outgoingEdges;
        for (auto itr = edgesFromInNode.begin(); itr != edgesFromInNode.end(); itr++)
        {
            if (*itr == edgeId)
            {
                edgesFromInNode.erase(itr);
                break;
            }
        }
    }
    {
        EdgeIds& edgesToOutNode = m_nodes[edge.getOutNode()].m_incomingEdges;
        for (auto itr = edgesToOutNode.begin(); itr != edgesToOutNode.end(); itr++)
        {
            if (*itr == edgeId)
            {
                edgesToOutNode.erase(itr);
                break;
            }
        }
    }

    // Remove the edge
    m_edges.erase(edgeId);

    assert(validate());
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::replaceEdgeId(EdgeId edgeId, EdgeId newId)
{
    assert(validate());
    assert(hasEdge(edgeId));
    assert(!hasEdge(newId));

    const Edge& edge = m_edges.at(edgeId);

    // Update incoming edges and outgoing edges of in/out nodes
    {
        EdgeIds& edgesFromInNode = m_nodes[edge.getInNode()].m_outgoingEdges;
        for (EdgeId& e : edgesFromInNode)
        {
            if (e == edgeId)
            {
                e = newId;
                break;
            }
        }
    }
    {
        EdgeIds& edgesToOutNode = m_nodes[edge.getOutNode()].m_incomingEdges;
        for (EdgeId& e : edgesToOutNode)
        {
            if (e == edgeId)
            {
                e = newId;
                break;
            }
        }
    }

    // Replace the edge id.
    m_edges[newId] = m_edges[edgeId];
    m_edges.erase(edgeId);

    assert(validate());
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::evaluate()
{
    assert(validate());

    // Initialize node evaluation states.
    for (auto& elem : m_nodes)
    {
        NodeData& nodeData = elem.second;
        nodeData.m_state = nodeData.getIncomingEdges().size() == 0 ? NodeData::EvalState::EVALUATED : NodeData::EvalState::NONE;
    }

    const bool circularNetwork = allowsCircularNetwork();

    std::unordered_set<NodeId> nodesInCurrentPath;
    std::vector<NodeId> stack;
    stack.reserve(4);

    // Evaluate all output nodes.
    for (NodeId outputNodeId : this->m_outputNodes)
    {
        stack.clear();
        nodesInCurrentPath.clear();
        stack.push_back(outputNodeId);
        while(stack.size() > 0)
        {
            NodeId id = stack.back();
            NodeData& node = m_nodes[id];

            if (node.m_state == NodeData::EvalState::EVALUATED)
            {
                stack.pop_back();
                continue;
            }

            assert(node.m_incomingEdges.size() > 0);

            // Calculate value of this node by visiting all parent nodes.
            float sumValue = 0;
            bool readyToEval = true;
            for (EdgeId incomingId : node.m_incomingEdges)
            {
                const Edge& edge = getEdge(incomingId);
                const float weight = edge.getWeight();

                if (weight == 0.f)
                {
                    continue;
                }

                NodeId inNodeId = edge.getInNode();

                bool isNewNode = true;
                if (circularNetwork)
                {
                    if (nodesInCurrentPath.find(inNodeId) != nodesInCurrentPath.end())
                    {
                        isNewNode = false;
                    }
                }

                const NodeData& inNodeData = m_nodes.at(inNodeId);

                // Recurse if we haven't evaluated this parent node yet.
                if (isNewNode && (inNodeData.m_state != NodeData::EvalState::EVALUATED))
                {
                    nodesInCurrentPath.insert(id);

                    stack.push_back(inNodeId);
                    readyToEval = false;
                    continue;
                }

                if (readyToEval)
                {
                    // Add a value from this parent.
                    sumValue += inNodeData.m_node.getValue() * weight;
                }
            }

            if (readyToEval)
            {
                assert(node.m_state != NodeData::EvalState::EVALUATED);
                // Set the node value and update its state.
                node.m_state = NodeData::EvalState::EVALUATED;
                node.m_node.setValue(sumValue);
                stack.pop_back();
                nodesInCurrentPath.erase(id);
            }
        }
    }
}

template <typename Node, typename Edge>
auto NeuralNetwork<Node, Edge>::bake() const->std::shared_ptr<BakedNeuralNetwork>
{
    return std::make_shared<BakedNeuralNetwork>(this);
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::validate() const
{
#ifdef DEBUG_SLOW
    if (m_nodes.size() < 2) return false;
    if (m_edges.size() < 1) return false;

    // Validate all edges.
    {
        std::unordered_set<EdgeId> edges;
        for (const auto& elem : m_edges)
        {
            // Make sure that the id is unique.
            EdgeId id = elem.first;
            if (edges.find(id) != edges.end()) return false;
            edges.insert(id);

            const Edge& e = elem.second;
            if (!hasNode(e.getInNode())) return false;
            if (!hasNode(e.getOutNode())) return false;
        }
    }

    // Validate all nodes.
    {
        int numInputOrBiasNode = 0;
        std::unordered_set<NodeId> nodes;
        for (const auto& elem : m_nodes)
        {
            const NodeData& nodeData = elem.second;

            // Make sure that the id is unique.
            NodeId id = nodeData.getId();
            if (nodes.find(id) != nodes.end()) return false;
            nodes.insert(id);

            std::unordered_set<EdgeId> edges;
            for (EdgeId e : getIncomingEdges(id))
            {
                if (!hasEdge(e)) return false;

                // Make sure that the id is unique.
                if (edges.find(e) != edges.end()) return false;
                edges.insert(e);
            }

            edges.clear();

            for (EdgeId e : getOutgoingEdges(id))
            {
                if (!hasEdge(e)) return false;

                // Make sure that the id is unique.
                if (edges.find(e) != edges.end()) return false;
                edges.insert(e);
            }
        }
    }
#endif
    return true;
}
