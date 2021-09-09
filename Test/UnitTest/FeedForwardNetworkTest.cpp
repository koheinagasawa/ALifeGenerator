/*
* FeedForwardNetworkTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>
#include <UnitTest/UnitTestBaseTypes.h>

#include <EvoAlgo/NeuralNetwork/FeedForwardNetwork.h>

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
        EXPECT_FALSE(nn.allowsCircularNetwork());
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
    EXPECT_TRUE(nn.isConnected(outNode, inNode));

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

    const float expectedValue = nodeVal1 * weight1 + nodeVal2 * weight2;

    // Evaluate
    nn.evaluate();
    EXPECT_TRUE(std::fabs((nn.getNode(outNode).getValue()) - expectedValue) < 1e-5f);

    // Evaluating multiple times shouldn't change the result for feed forward network.
    nn.evaluate();
    EXPECT_TRUE(std::fabs((nn.getNode(outNode).getValue()) - expectedValue) < 1e-5f);
}

TEST(FeedForwardNetwork, AddEdge)
{
    // Set up node and edges.
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode1(2);
    NodeId outNode2(3);
    NodeId hiddenNode1(4);
    NodeId hiddenNode2(5);

    FFN::Nodes nodes;
    nodes.insert({ inNode1, Node() });
    nodes.insert({ inNode2, Node() });
    nodes.insert({ outNode1, Node() });
    nodes.insert({ outNode2, Node() });
    nodes.insert({ hiddenNode1, Node() });
    nodes.insert({ hiddenNode2, Node() });

    EdgeId edge1(1);
    EdgeId edge2(2);
    EdgeId edge3(3);
    EdgeId edge4(4);

    FFN::Edges edges;
    edges.insert({ edge1, Edge(inNode1, hiddenNode1, 0.5f) });
    edges.insert({ edge2, Edge(inNode2, hiddenNode2, 0.5f) });
    edges.insert({ edge3, Edge(hiddenNode1, outNode1, 0.5f) });
    edges.insert({ edge4, Edge(hiddenNode2, outNode2, 0.5f) });

    FFN::NodeIds inputNodes;
    inputNodes.push_back(inNode1);
    inputNodes.push_back(inNode2);
    FFN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    // Create a NeuralNetwork.
    FFN ffn(nodes, edges, inputNodes, outputNodes);

    EXPECT_TRUE(ffn.validate());
    EXPECT_EQ(ffn.getNumNodes(), 6);
    int numEdges = 4;
    EXPECT_EQ(ffn.getNumEdges(), numEdges);

    // Add an edge.
    EdgeId edge5(5);
    EXPECT_TRUE(ffn.addEdgeAt(inNode1, hiddenNode2, edge5, 0.1f));
    EXPECT_TRUE(ffn.hasEdge(edge5));
    EXPECT_EQ(ffn.getNumEdges(), ++numEdges);
    EXPECT_EQ(ffn.getWeight(edge5), 0.1f);
    EXPECT_EQ(ffn.getInNode(edge5), inNode1);
    EXPECT_EQ(ffn.getOutNode(edge5), hiddenNode2);
    EXPECT_EQ(ffn.getIncomingEdges(hiddenNode2).size(), 2);
    EXPECT_EQ(ffn.getIncomingEdges(hiddenNode2)[0], edge2);
    EXPECT_EQ(ffn.getIncomingEdges(hiddenNode2)[1], edge5);

    // Try to add an edge at nodes which are already connected.
    {
        EdgeId e(6);
        EXPECT_FALSE(ffn.addEdgeAt(inNode1, hiddenNode1, e, 0.5f));
        EXPECT_EQ(ffn.getNumEdges(), numEdges);
        EXPECT_FALSE(ffn.hasEdge(e));
    }

    // Try to add an edge going from an outputNode.
    {
        EdgeId e(6);
        EXPECT_FALSE(ffn.addEdgeAt(outNode1, inNode2, e, 0.1f));
        EXPECT_EQ(ffn.getNumEdges(), numEdges);
        EXPECT_FALSE(ffn.hasEdge(e));
        EXPECT_FALSE(ffn.addEdgeAt(outNode2, hiddenNode1, e, 0.1f));
        EXPECT_EQ(ffn.getNumEdges(), numEdges);
        EXPECT_FALSE(ffn.hasEdge(e));
    }

    // Add an edge going into an inputNode.
    {
        EdgeId edge6(6);
        EXPECT_FALSE(ffn.addEdgeAt(inNode1, inNode2, edge6, 0.2f));
        EXPECT_FALSE(ffn.hasEdge(edge6));
        EXPECT_EQ(ffn.getNumEdges(), numEdges);
        EXPECT_EQ(ffn.getIncomingEdges(inNode2).size(), 0);
    }

    // Try to add an edge at a node which doesn't exit.
    {
        EdgeId e(7);
        EXPECT_FALSE(ffn.addEdgeAt(hiddenNode1, NodeId(6), e, 0.1f));
        EXPECT_EQ(ffn.getNumEdges(), numEdges);
        EXPECT_FALSE(ffn.hasEdge(e));
        EXPECT_FALSE(ffn.addEdgeAt(NodeId(7), outNode1, e, 0.1f));
        EXPECT_EQ(ffn.getNumEdges(), numEdges);
        EXPECT_FALSE(ffn.hasEdge(e));
    }

    // Try to add an edge which creates a circle.
    {
        EdgeId e(7);
        EXPECT_TRUE(ffn.addEdgeAt(hiddenNode1, hiddenNode2, e, 0.1f));
        EXPECT_EQ(ffn.getNumEdges(), ++numEdges);
        EXPECT_TRUE(ffn.hasEdge(e));
        EXPECT_EQ(ffn.getIncomingEdges(hiddenNode2).size(), 3);
        EdgeId e2(8);
        EXPECT_FALSE(ffn.addEdgeAt(hiddenNode2, hiddenNode1, e2, 0.1f));
        EXPECT_EQ(ffn.getNumEdges(), numEdges);
        EXPECT_FALSE(ffn.hasEdge(e2));
        EXPECT_EQ(ffn.getIncomingEdges(hiddenNode1).size(), 1);
    }
}
