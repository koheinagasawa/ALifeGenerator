/*
* FeedForwardNetworkTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/NeuralNetwork/FeedForwardNetwork.h>

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

using FFN = FeedForwardNetwork<Node, Edge>;

TEST(FeedForwardNetwork, CreateInvalidNetworks)
{
    FFN::Nodes nodes;
    FFN::Edges edges;
    FFN::NodeIds inputNodes;
    FFN::NodeIds outputNodes;

    // Empty network
    {
        FFN nn(nodes, edges, inputNodes, outputNodes);
        EXPECT_FALSE(nn.validate());
    }

    NodeId inNode(0);
    NodeId outNode(1);

    nodes.insert({ inNode, Node() });
    nodes.insert({ outNode, Node() });

    edges.insert({ EdgeId(0), Edge(inNode, outNode) });

    outputNodes.push_back(outNode);

    // Network with no input node
    {
        FFN nn(nodes, edges, inputNodes, outputNodes);
        EXPECT_FALSE(nn.validate());
    }

    outputNodes.clear();
    inputNodes.push_back(inNode);

    // Network with no output node
    {
        FFN nn(nodes, edges, inputNodes, outputNodes);
        EXPECT_FALSE(nn.validate());
    }

    outputNodes.push_back(outNode);

    // Invalid edge
    {
        FFN::Edges edges2 = edges;
        edges2.insert({ EdgeId(1), Edge(NodeId(2), NodeId(3)) });

        FFN nn(nodes, edges2, inputNodes, outputNodes);
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

        FFN nn(nodes, edges, inputNodes, outputNodes);
        EXPECT_FALSE(nn.validate());
    }
}

TEST(FeedForwardNetwork, CreateMinimumNetwork)
{
    // Set up node and edges.
    NodeId inNode(0);
    NodeId outNode(1);

    FFN::Nodes nodes;
    nodes.insert({ inNode, Node() });
    nodes.insert({ outNode, Node() });

    EdgeId edge(0);

    FFN::Edges edges;
    edges.insert({ edge, Edge(inNode, outNode) });

    FFN::NodeIds inputNodes;
    inputNodes.push_back(inNode);

    FFN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    // Create a network.
    FFN nn(nodes, edges, inputNodes, outputNodes);

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

    EXPECT_EQ(nn.getInputNodes().size(), 1);
    EXPECT_EQ(nn.getInputNodes()[0], inNode);
    EXPECT_EQ(nn.getOutputNodes().size(), 1);
    EXPECT_EQ(nn.getOutputNodes()[0], outNode);
}

TEST(FeedForwardNetwork, EvaluateSimpleNetwork)
{
    // Set up node and edges.
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode(2);
    float nodeVal1 = 5.f, nodeVal2 = 7.f;

    FFN::Nodes nodes;
    {
        nodes.insert({ inNode1, Node(nodeVal1) });
        nodes.insert({ inNode2, Node(nodeVal2) });
        nodes.insert({ outNode, Node() });
    }

    EdgeId edgeId1(0);
    EdgeId edgeId2(1);
    float weight1 = 0.5f, weight2 = 0.3f;

    FFN::Edges edges;
    {
        Edge edge1(inNode1, outNode);
        edge1.m_weight = weight1;
        edges.insert({ edgeId1, edge1 });
        Edge edge2(inNode2, outNode);
        edge2.m_weight = weight2;
        edges.insert({ edgeId2, edge2 });
    }

    FFN::NodeIds inputNodes;
    inputNodes.push_back(inNode1);
    inputNodes.push_back(inNode2);

    FFN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    // Create a network.
    FFN nn(nodes, edges, inputNodes, outputNodes);

    EXPECT_EQ(nn.getNumNodes(), 3);
    EXPECT_EQ(nn.getNumEdges(), 2);

    EXPECT_TRUE(nn.validate());

    nn.evaluate();

    EXPECT_TRUE(std::abs((nn.getNode(outNode).getValue()) - (nodeVal1 * weight1 + nodeVal2 * weight2)) < 1e-5f);
}
