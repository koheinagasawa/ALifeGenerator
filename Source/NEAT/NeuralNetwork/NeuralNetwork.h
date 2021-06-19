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

// Type of neural network
enum class NeuralNetworkType
{
    GENERAL,
    FEED_FORWARD,
    RECURRENT,
};

// Base class of neural network
template <typename Node, typename Edge>
class NeuralNetwork
{
public:
    //
    // Type Declarations
    //

    using Nodes = std::unordered_map<NodeId, Node>;
    using Edges = std::unordered_map<EdgeId, Edge>;
    using NodeIds = std::vector<NodeId>;
    using EdgeIds = std::vector<EdgeId>;

    // Node and some additional data for shortcut access.
    struct NodeData
    {
        Node m_node;
        EdgeIds m_incomingEdges;
    };

    using NodeDatas = std::unordered_map<NodeId, NodeData>;
    using NodeEntry = std::pair<NodeId, NodeData>;
    using EdgeEntry = std::pair<EdgeId, Edge>;

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

    // Return true if the node id exists.
    inline bool hasNode(NodeId id) const { return m_nodes.find(id) != m_nodes.end(); }

    // Get read-only access to the node of given id.
    inline auto getNode(NodeId id) const->const Node&;

    // Get read-write access to the node of given id.
    inline auto accessNode(NodeId id)->Node&;

    // Get a list of incoming edges of the node.
    inline auto getIncomingEdges(NodeId id) const->EdgeIds;

    // Return true if the two nodes are connected.
    inline bool isConnected(NodeId node1, NodeId node2) const;

    // Set a value of the node.
    inline void setNodeValue(NodeId id, float value);

    // Get input node ids.
    inline auto getInputNodes() const->NodeIds { return m_inputNodes; }

    // Get output node ids.
    inline auto getOutputNodes() const->NodeIds { return m_outputNodes; }

    //
    // Edge Queries
    //

    // Return the number of edges.
    inline int getNumEdges() const { return (int)m_edges.size(); }

    // Return all the edges in the network.
    inline auto getEdges() const->const Edges& { return m_edges; }

    // Return true if the edge id exists.
    inline bool hasEdge(EdgeId id) const { return m_edges.find(id) != m_edges.end(); }

    // Return read-only access to an edge in the network.
    inline auto getEdge(EdgeId id) const->const Edge& { return m_edges.at(id); }

    // Return read-write access to an edge in the network.
    inline auto accessEdge(EdgeId id)->Edge& { return m_edges[id]; }

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
    virtual bool addNodeAt(EdgeId edgeId, NodeId newNodeId, EdgeId newIncomingEdgeId, EdgeId newOutgoingEdgeId);

    // Add a new edge between node1 and node2 with weight.
    virtual bool addEdgeAt(NodeId node1, NodeId node2, EdgeId newEdgeId, float weight = 1.0f);

    // Remove an existing edge
    virtual void removeEdge(EdgeId edgeId);

    // Replace an node id with a new node id.
    virtual void replaceNodeId(NodeId nodeId, NodeId newId);

    // Replace an edge id with a new edge id.
    virtual void replaceEdgeId(EdgeId edgeId, EdgeId newId);

    // Return true if a new edge can be added between inNode and outNode.
    virtual bool canAddEdgeAt(NodeId inNode, NodeId outNode) const { return true; }

    //
    // Evaluation
    //

    virtual void evaluate() {}

    //
    // Validation
    //

    // Return false if this network has invalid data.
    virtual bool validate() const;

protected:
    // Default constructor.
    NeuralNetwork() = default;

    // Construct m_nodes. This is called from constructor.
    void constructNodeData(const Nodes& nodes);

    NodeDatas m_nodes; // Nodes of this network.
    Edges m_edges; // Edges of this network.

    NodeIds m_inputNodes; // A list of output nodes of this network.
    NodeIds m_outputNodes; // A list of output nodes of this network.
};

template <typename Node, typename Edge>
NeuralNetwork<Node, Edge>::NeuralNetwork(const Nodes& nodes, const Edges& edges)
    : m_edges(edges)
{
    constructNodeData(nodes);
}

template <typename Node, typename Edge>
NeuralNetwork<Node, Edge>::NeuralNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& inputNodes, const NodeIds& outputNodes)
    : m_edges(edges)
    , m_inputNodes(inputNodes)
    , m_outputNodes(outputNodes)
{
    constructNodeData(nodes);
}

template <typename Node, typename Edge>
auto NeuralNetwork<Node, Edge>::clone() const->std::shared_ptr<NeuralNetwork<Node, Edge>>
{
    return std::make_shared<NeuralNetwork>(*this);
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::constructNodeData(const Nodes& nodes)
{
    // Allocate NodeDatas
    m_nodes.clear();
    m_nodes.reserve(nodes.size());
    for (const auto& itr : nodes)
    {
        m_nodes.insert({ itr.first, NodeData{ itr.second } });
    }

    // Set incoming edges array in each node
    for (const auto& itr : m_edges)
    {
        const Edge& e = itr.second;
        NodeId outNode = e.getOutNode();
        if (!hasNode(outNode))
        {
            WARN("Input edge contains invalid outNode value.");
            continue;
        }

        m_nodes[outNode].m_incomingEdges.push_back(itr.first);
    }
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
    return m_nodes[id].m_node;
}

template <typename Node, typename Edge>
inline void NeuralNetwork<Node, Edge>::setNodeValue(NodeId id, float value)
{
    if (hasNode(id))
    {
        m_nodes[id].m_node.setValue(value);
    }
    else
    {
        WARN("Trying to set a value for a node which doesn't exist.");
    }
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getIncomingEdges(NodeId id) const->EdgeIds
{
    return m_nodes.at(id).m_incomingEdges;
}

template <typename Node, typename Edge>
inline bool NeuralNetwork<Node, Edge>::isConnected(NodeId node1, NodeId node2) const
{
    assert(hasNode(node1) && hasNode(node2) && node1 != node2);

    for (EdgeId e : m_nodes.at(node1).m_incomingEdges)
    {
        if (getInNode(e) == node2)
        {
            return true;
        }
    }

    for (EdgeId e : m_nodes.at(node2).m_incomingEdges)
    {
        if (getInNode(e) == node1)
        {
            return true;
        }
    }

    return false;
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
    const Edge& e = m_edges.at(id);
    return e.getInNode();
}

template <typename Node, typename Edge>
inline auto NeuralNetwork<Node, Edge>::getOutNode(EdgeId id) const->NodeId
{
    assert(hasEdge(id));
    return m_edges.at(id).getOutNode();
}

template <typename Node, typename Edge>
bool NeuralNetwork<Node, Edge>::addNodeAt(EdgeId edgeId, NodeId newNodeId, EdgeId newIncomingEdgeId, EdgeId newOutgoingEdgeId)
{
    assert(validate());
    assert(!hasNode(newNodeId) && !hasEdge(newIncomingEdgeId) && !hasEdge(newOutgoingEdgeId));

    // Make sure that edgeId exists.
    if (!hasEdge(edgeId))
    {
        WARN("Edge id doesn't exist.");
        return false;
    }

    // Disable the divided edge
    const Edge& edgeToDivide = m_edges.at(edgeId);

    // Create two new edges.
    Edge newEdge1(edgeToDivide.getInNode(), newNodeId, 1.0f);
    {
        m_edges.insert({ newIncomingEdgeId, newEdge1 });
    }
    Edge newEdge2(newNodeId, edgeToDivide.getOutNode(), 1.0f);
    {
        m_edges.insert({ newOutgoingEdgeId, newEdge2 });
    }

    // Create a new node.
    NodeData newNode;
    newNode.m_incomingEdges.push_back(newIncomingEdgeId);
    m_nodes.insert({ newNodeId, newNode });

    // Update incoming edges of the out node.
    m_nodes[edgeToDivide.getOutNode()].m_incomingEdges.push_back(newOutgoingEdgeId);

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

    NodeData& outNodeData = m_nodes[node2];
    EdgeIds& edgesToOutNode = outNodeData.m_incomingEdges;

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
    m_edges.insert({ newEdgeId, Edge(node1, node2, weight) });
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
    for (auto& itr : m_edges)
    {
        const Edge& curEdge = itr.second;
        if (curEdge.getInNode() == nodeId)
        {
            Edge newEdge(newId, curEdge.getOutNode());
            newEdge.copyState(&curEdge);
            itr.second = newEdge;
        }
        else if (curEdge.getOutNode() == nodeId)
        {
            Edge newEdge(curEdge.getInNode(), newId);
            newEdge.copyState(&curEdge);
            itr.second = newEdge;
        }
    }

    // Replace the node itself
    const NodeData& nd = m_nodes.at(nodeId);
    m_nodes.insert({ newId, nd });
    m_nodes.erase(nodeId);

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

    // Update incoming edges of output node
    NodeData& outputNode = m_nodes[edge.getOutNode()];
    EdgeIds& edgesToOutNode = outputNode.m_incomingEdges;
    for (auto itr = edgesToOutNode.begin(); itr != edgesToOutNode.end(); itr++)
    {
        if (*itr == edgeId)
        {
            edgesToOutNode.erase(itr);
            break;
        }
    }

    // Remove the original edge id
    for (auto itr = m_edges.begin(); itr != m_edges.end(); itr++)
    {
        if (itr->first == edgeId)
        {
            m_edges.erase(itr);
            break;
        }
    }

    assert(validate());
}

template <typename Node, typename Edge>
void NeuralNetwork<Node, Edge>::replaceEdgeId(EdgeId edgeId, EdgeId newId)
{
    assert(validate());
    assert(hasEdge(edgeId));
    assert(!hasEdge(newId));

    const Edge& edge = m_edges.at(edgeId);

    // Update incoming edges of output node
    NodeData& outputNode = m_nodes[edge.getOutNode()];
    EdgeIds& edgesToOutNode = outputNode.m_incomingEdges;
    for (EdgeId& e : edgesToOutNode)
    {
        if (e == edgeId)
        {
            e = newId;
            break;
        }
    }

    // Replace the edge.
    const Edge e = m_edges.at(edgeId);

    // Remove the original edge id
    for (auto itr = m_edges.begin(); itr != m_edges.end(); itr++)
    {
        if (itr->first == edgeId)
        {
            m_edges.erase(itr);
            break;
        }
    }

    // Add the new edge id
    m_edges.insert({ newId, e });

    assert(validate());
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
        for (const auto& itr : m_edges)
        {
            // Make sure that the id is unique.
            EdgeId id = itr.first;
            if (edges.find(id) != edges.end()) return false;
            edges.insert(id);

            const Edge& e = itr.second;
            if (!hasNode(e.getInNode())) return false;
            if (!hasNode(e.getOutNode())) return false;
        }
    }

    // Validate all nodes.
    {
        int numInputOrBiasNode = 0;
        std::unordered_set<NodeId> nodes;
        for (const auto& itr : m_nodes)
        {
            // Make sure that the id is unique.
            NodeId id = itr.first;
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
        }
    }
#endif
    return true;
}
