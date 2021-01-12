/*
* NeuralNetworkTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/NeuralNetwork.h>

// Basic node class.
struct Node : public NodeBase
{
    Node() = default;
    Node(float value) : m_value(value) {}

    virtual float getValue() const { return m_value; }
    virtual void setValue(float value) { m_value = value; }

    float m_value;
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

    NodeId m_inNode, m_outNode;
    float m_weight;
};

using NN = NeuralNetwork<Node, Edge>;

TEST(NeuralNetwork, CreateInvalidNetworks)
{
    NN::Nodes nodes;
    NN::Edges edges;
    NN::NodeIds outputNodes;

    NN nn(nodes, edges, outputNodes);
    EXPECT_FALSE(nn.validate());

    NodeId inNode(0);
    NodeId outNode(1);

    nodes[inNode] = Node();
    nodes[outNode] = Node();

    edges[EdgeId(0)] = Edge(inNode, outNode);

    NN nn2(nodes, edges, outputNodes);
    EXPECT_FALSE(nn2.validate());

    outputNodes.push_back(outNode);

    NN::Edges edges2 = edges;
    edges2[EdgeId(1)] = Edge(NodeId(2), NodeId(3));

    NN nn3(nodes, edges2, outputNodes);
    EXPECT_FALSE(nn3.validate());
}

TEST(NeuralNetwork, CreateMinimumNetwork)
{
    NodeId inNode(0);
    NodeId outNode(1);

    NN::Nodes nodes;
    nodes[inNode] = Node();
    nodes[outNode] = Node();

    EdgeId edge(0);

    NN::Edges edges;
    edges[edge] = Edge(inNode, outNode);

    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    NN nn(nodes, edges, outputNodes);

    EXPECT_TRUE(nn.validate());

    EXPECT_TRUE(nn.hasNode(inNode));
    EXPECT_TRUE(nn.hasNode(outNode));
    EXPECT_FALSE(nn.hasNode(NodeId(2)));

    NN::NodeIds nodeIds = nn.getNodes();
    EXPECT_EQ(nodeIds.size(), 2);

    EXPECT_TRUE(nn.hasEdge(edge));
    EXPECT_FALSE(nn.hasEdge(EdgeId(1)));
    EXPECT_EQ(nn.getInNode(edge), inNode);
    EXPECT_EQ(nn.getOutNode(edge), outNode);

    NN::NodeIds ons = nn.getOutputNodes();
    EXPECT_EQ(ons.size(), 1);
}

TEST(NeuralNetwork, GetSetNodeValues)
{
    NodeId inNode(0);
    NodeId outNode(1);

    NN::Nodes nodes;
    nodes[inNode] = Node(5.f);
    nodes[outNode] = Node(7.f);

    EdgeId edge(0);

    NN::Edges edges;
    edges[edge] = Edge(inNode, outNode);

    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    NN nn(nodes, edges, outputNodes);

    EXPECT_TRUE(nn.validate());

    EXPECT_EQ(nn.getNode(inNode).getValue(), 5.f);
    EXPECT_EQ(nn.getNode(outNode).getValue(), 7.f);
    nn.setNodeValue(inNode, 3.f);
    EXPECT_EQ(nn.getNode(inNode).getValue(), 3.f);
}

TEST(NeuralNetwork, GetSetEdgeWeights)
{
    NodeId inNode(0);
    NodeId outNode(1);

    NN::Nodes nodes;
    nodes[inNode] = Node();
    nodes[outNode] = Node();

    EdgeId edgeId(0);

    NN::Edges edges;
    Edge edge(inNode, outNode);
    edge.m_weight = 10.f;
    edges[edgeId] = edge;

    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    NN nn(nodes, edges, outputNodes);

    EXPECT_TRUE(nn.validate());

    EXPECT_EQ(nn.getWeight(edgeId), 10.f);
    nn.setWeight(edgeId, 12.f);
    EXPECT_EQ(nn.getWeight(edgeId), 12.f);
}

TEST(NeuralNetwork, EvaluateSimpleNetwork)
{
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode(2);
    float nodeVal1 = 5.f, nodeVal2 = 7.f;

    NN::Nodes nodes;
    {
        nodes[inNode1] = Node(nodeVal1);
        nodes[inNode2] = Node(nodeVal2);
        nodes[outNode] = Node();
    }

    EdgeId edgeId1(0);
    EdgeId edgeId2(1);
    float weight1 = 0.5f, weight2 = 0.3f;

    NN::Edges edges;
    {
        Edge edge1(inNode1, outNode);
        edge1.m_weight = weight1;
        edges[edgeId1] = edge1;
        Edge edge2(inNode2, outNode);
        edge2.m_weight = weight2;
        edges[edgeId2] = edge2;
    }

    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    NN nn(nodes, edges, outputNodes);

    EXPECT_TRUE(nn.validate());

    nn.evaluate();

    EXPECT_EQ(nn.getNode(outNode).getValue(), nodeVal1 * weight1 + nodeVal2 * weight2); 
}
