/*
* NeuralNetworkTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/NeuralNetwork/NeuralNetwork.h>

// Basic node class.
struct Node : public NodeBase
{
    Node() = default;
    Node(float value) : m_value(value) {}

    virtual float getValue() const { return m_value; }
    virtual void setValue(float value) { m_value = value; }

    float m_value = 0.f;
};

// Basic edge class.
struct Edge : public EdgeBase
{
    Edge() = default;
    Edge(NodeId inNode, NodeId outNode) : m_inNode(inNode), m_outNode(outNode) {}

    virtual NodeId getInNode() const { return m_inNode; }
    virtual NodeId getOutNode() const { return m_outNode; }
    virtual float getWeight() const { return m_weight; }
    virtual void setWeight(float weight) { m_weight = weight; }

    NodeId m_inNode = NodeId::invalid();
    NodeId m_outNode = NodeId::invalid();
    float m_weight = 0.f;
};

using NN = NeuralNetwork<Node, Edge>;

TEST(NeuralNetwork, CreateInvalidNetworks)
{
    NN::Nodes nodes;
    NN::Edges edges;

    // Empty network
    {
        NN nn(nodes, edges);
        EXPECT_FALSE(nn.validate());
    }

    NodeId inNode(0);
    NodeId outNode(1);

    nodes.insert({ inNode, Node() });
    nodes.insert({ outNode, Node() });

    edges.insert({ EdgeId(0), Edge(inNode, outNode) });

    // Invalid edge
    {
        NN::Edges edges2 = edges;
        edges2.insert({ EdgeId(1), Edge(NodeId(2), NodeId(3)) });

        NN nn(nodes, edges2);
        EXPECT_FALSE(nn.validate());
    }

    // Circular network
    {
        NodeId node1(2);
        NodeId node2(3);
        NodeId node3(4);
        nodes.insert({ node1, Node() });
        nodes.insert({ node2, Node() });
        nodes.insert({ node3, Node() });

        edges.insert({ EdgeId(1), Edge(inNode, node1) });
        edges.insert({ EdgeId(2), Edge(node1, node2) });
        edges.insert({ EdgeId(3), Edge(node2, node3) });
        edges.insert({ EdgeId(4), Edge(node3, node1) });
        edges.insert({ EdgeId(5), Edge(node3, outNode) });

        NN nn(nodes, edges);
        EXPECT_TRUE(nn.validate());
    }
}

TEST(NeuralNetwork, CreateMinimumNetwork)
{
    // Set up node and edges.
    NodeId inNode(0);
    NodeId outNode(1);

    NN::Nodes nodes;
    nodes.insert({ inNode, Node() });
    nodes.insert({ outNode, Node() });

    EdgeId edge(0);

    NN::Edges edges;
    edges.insert({ edge, Edge(inNode, outNode) });

    // Create a network.
    NN nn(nodes, edges);

    EXPECT_TRUE(nn.validate());

    EXPECT_TRUE(nn.hasNode(inNode));
    EXPECT_TRUE(nn.hasNode(outNode));
    EXPECT_FALSE(nn.hasNode(NodeId(2)));

    EXPECT_EQ(nn.getIncomingEdges(inNode).size(), 0);
    EXPECT_EQ(nn.getIncomingEdges(outNode).size(), 1);
    EXPECT_EQ(nn.getIncomingEdges(outNode)[0], edge);
    EXPECT_TRUE(nn.isConnected(inNode, outNode));

    EXPECT_EQ(nn.getNumNodes(), 2);
    EXPECT_EQ(nn.getNumEdges(), 1);

    EXPECT_TRUE(nn.hasEdge(edge));
    EXPECT_FALSE(nn.hasEdge(EdgeId(1)));
    EXPECT_EQ(nn.getInNode(edge), inNode);
    EXPECT_EQ(nn.getOutNode(edge), outNode);
}

TEST(NeuralNetwork, GetSetNodeValues)
{
    // Set up node and edges.
    NodeId inNode(0);
    NodeId outNode(1);

    NN::Nodes nodes;
    nodes.insert({ inNode, Node(5.f) });
    nodes.insert({ outNode, Node(7.f) });

    EdgeId edge(0);

    NN::Edges edges;
    edges.insert({ edge, Edge(inNode, outNode) });

    // Create a network.
    NN nn(nodes, edges);

    EXPECT_TRUE(nn.validate());

    EXPECT_EQ(nn.getNode(inNode).getValue(), 5.f);
    EXPECT_EQ(nn.getNode(outNode).getValue(), 7.f);
    nn.setNodeValue(inNode, 3.f);
    EXPECT_EQ(nn.getNode(inNode).getValue(), 3.f);
}

TEST(NeuralNetwork, GetSetEdgeWeights)
{
    // Set up node and edges.
    NodeId inNode(0);
    NodeId outNode(1);

    NN::Nodes nodes;
    nodes.insert({ inNode, Node() });
    nodes.insert({ outNode, Node() });

    EdgeId edgeId(0);

    NN::Edges edges;
    Edge edge(inNode, outNode);
    edge.m_weight = 10.f;
    edges.insert({ edgeId, edge });

    // Create a network.
    NN nn(nodes, edges);

    EXPECT_TRUE(nn.validate());

    EXPECT_EQ(nn.getWeight(edgeId), 10.f);
    nn.setWeight(edgeId, 12.f);
    EXPECT_EQ(nn.getWeight(edgeId), 12.f);
}
