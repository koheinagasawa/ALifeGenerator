/*
* NeuralNetworkTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>
#include <UnitTest/UnitTestBaseTypes.h>

#include <NEAT/NeuralNetwork/NeuralNetwork.h>

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
}

TEST(NeuralNetwork, CreateCircularNetwork)
{
    NN::Nodes nodes;
    NN::Edges edges;

    NodeId inNode(0);
    NodeId outNode(1);

    nodes.insert({ inNode, Node() });
    nodes.insert({ outNode, Node() });
    edges.insert({ EdgeId(0), Edge(inNode, outNode) });

    // Circular network
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
    EXPECT_TRUE(nn.allowsCircularNetwork());
    EXPECT_TRUE(nn.validate());
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

    {
        // Create a network.
        NN nn(nodes, edges);

        EXPECT_TRUE(nn.validate());

        EXPECT_TRUE(nn.hasNode(inNode));
        EXPECT_TRUE(nn.hasNode(outNode));
        EXPECT_FALSE(nn.hasNode(NodeId(2)));

        EXPECT_EQ(nn.getIncomingEdges(inNode).size(), 0);
        EXPECT_EQ(nn.getIncomingEdges(outNode).size(), 1);
        EXPECT_EQ(nn.getIncomingEdges(outNode)[0], edge);
        EXPECT_EQ(nn.getOutgoingEdges(inNode).size(), 1);
        EXPECT_EQ(nn.getOutgoingEdges(inNode)[0], edge);
        EXPECT_EQ(nn.getOutgoingEdges(outNode).size(), 0);
        EXPECT_TRUE(nn.isConnected(inNode, outNode));
        EXPECT_TRUE(nn.isConnected(outNode, inNode));

        EXPECT_EQ(nn.getNumNodes(), 2);
        EXPECT_EQ(nn.getNumEdges(), 1);

        EXPECT_TRUE(nn.hasEdge(edge));
        EXPECT_FALSE(nn.hasEdge(EdgeId(1)));
        EXPECT_EQ(nn.getInNode(edge), inNode);
        EXPECT_EQ(nn.getOutNode(edge), outNode);

        // Input and output nodes are empty unless they are specified explicitly by constructor.
        EXPECT_EQ(nn.getInputNodes().size(), 0);
        EXPECT_EQ(nn.getOutputNodes().size(), 0);
    }

    {
        // Create a network with input and output nodes.
        std::vector<NodeId> inputNodes;
        inputNodes.push_back(inNode);
        std::vector<NodeId> outputNodes;
        outputNodes.push_back(outNode);
        NN nn(nodes, edges, inputNodes, outputNodes);

        EXPECT_TRUE(nn.validate());

        EXPECT_TRUE(nn.hasNode(inNode));
        EXPECT_TRUE(nn.hasNode(outNode));
        EXPECT_FALSE(nn.hasNode(NodeId(2)));

        EXPECT_EQ(nn.getIncomingEdges(inNode).size(), 0);
        EXPECT_EQ(nn.getIncomingEdges(outNode).size(), 1);
        EXPECT_EQ(nn.getIncomingEdges(outNode)[0], edge);
        EXPECT_EQ(nn.getOutgoingEdges(inNode).size(), 1);
        EXPECT_EQ(nn.getOutgoingEdges(inNode)[0], edge);
        EXPECT_EQ(nn.getOutgoingEdges(outNode).size(), 0);
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

TEST(NeuralNetwork, AddNode)
{
    // Set up node and edges.
    NodeId inNode(0);
    NodeId outNode(1);

    NN::Nodes nodes;
    nodes.insert({ inNode, Node() });
    nodes.insert({ outNode, Node() });

    EdgeId edge(0);

    NN::Edges edges;
    edges.insert({ edge, Edge(inNode, outNode, 0.5f) });

    NN::NodeIds inputNodes;
    inputNodes.push_back(inNode);
    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    // Create a NeuralNetwork.
    NN nn(nodes, edges, inputNodes, outputNodes);

    EXPECT_TRUE(nn.validate());
    EXPECT_EQ(nn.getNumNodes(), 2);
    EXPECT_EQ(nn.getNumEdges(), 1);
    EXPECT_EQ(nn.getWeight(edge), 0.5f);

    // Try to add a node at an edge which doesn't exist.
    NodeId newNode(2);
    EdgeId newIncomingEdge(1);
    EdgeId newOutgoingEdge(2);
    nn.addNodeAt(EdgeId(1), newNode, newIncomingEdge, newOutgoingEdge);
    EXPECT_EQ(nn.getNumNodes(), 2);
    EXPECT_EQ(nn.getNumEdges(), 1);

    // Add one node
    nn.addNodeAt(edge, newNode, newIncomingEdge, newOutgoingEdge);

    EXPECT_TRUE(nn.hasNode(inNode));
    EXPECT_TRUE(nn.hasNode(outNode));
    EXPECT_TRUE(nn.hasNode(newNode));
    EXPECT_TRUE(nn.hasEdge(edge));
    EXPECT_TRUE(nn.hasEdge(newIncomingEdge));
    EXPECT_TRUE(nn.hasEdge(newOutgoingEdge));
    EXPECT_EQ(nn.getWeight(edge), 0.5f);
    EXPECT_EQ(nn.getWeight(newIncomingEdge), 1.0f);
    EXPECT_EQ(nn.getWeight(newOutgoingEdge), 1.0f);
    EXPECT_EQ(nn.getNumNodes(), 3);
    EXPECT_EQ(nn.getNumEdges(), 3);
    EXPECT_EQ(nn.getInNode(edge), inNode);
    EXPECT_EQ(nn.getOutNode(edge), outNode);
    EXPECT_EQ(nn.getInNode(newIncomingEdge), inNode);
    EXPECT_EQ(nn.getOutNode(newIncomingEdge), newNode);
    EXPECT_EQ(nn.getInNode(newOutgoingEdge), newNode);
    EXPECT_EQ(nn.getOutNode(newOutgoingEdge), outNode);
    EXPECT_EQ(nn.getIncomingEdges(inNode).size(), 0);
    EXPECT_EQ(nn.getIncomingEdges(newNode).size(), 1);
    EXPECT_EQ(nn.getIncomingEdges(newNode)[0], newIncomingEdge);
    EXPECT_EQ(nn.getIncomingEdges(outNode).size(), 2);
    EXPECT_EQ(nn.getIncomingEdges(outNode)[0], edge);
    EXPECT_EQ(nn.getIncomingEdges(outNode)[1], newOutgoingEdge);
    EXPECT_EQ(nn.getOutgoingEdges(inNode).size(), 2);
    EXPECT_EQ(nn.getOutgoingEdges(inNode)[0], edge);
    EXPECT_EQ(nn.getOutgoingEdges(inNode)[1], newIncomingEdge);
    EXPECT_EQ(nn.getOutgoingEdges(newNode).size(), 1);
    EXPECT_EQ(nn.getOutgoingEdges(newNode)[0], newOutgoingEdge);
    EXPECT_EQ(nn.getOutgoingEdges(outNode).size(), 0);

    // Add one more node
    NodeId newNode2(3);
    EdgeId newIncomingEdge2(3), newOutgoingEdge2(4);
    nn.addNodeAt(newOutgoingEdge, newNode2, newIncomingEdge2, newOutgoingEdge2);

    EXPECT_TRUE(nn.hasNode(inNode));
    EXPECT_TRUE(nn.hasNode(outNode));
    EXPECT_TRUE(nn.hasNode(newNode));
    EXPECT_TRUE(nn.hasNode(newNode2));
    EXPECT_TRUE(nn.hasEdge(edge));
    EXPECT_TRUE(nn.hasEdge(newOutgoingEdge));
    EXPECT_TRUE(nn.hasEdge(newIncomingEdge2));
    EXPECT_TRUE(nn.hasEdge(newOutgoingEdge2));
    EXPECT_EQ(nn.getWeight(newIncomingEdge), 1.f);
    EXPECT_EQ(nn.getWeight(newIncomingEdge2), 1.f);
    EXPECT_EQ(nn.getWeight(newOutgoingEdge2), 1.f);
    EXPECT_EQ(nn.getNumNodes(), 4);
    EXPECT_EQ(nn.getNumEdges(), 5);
    EXPECT_EQ(nn.getInNode(edge), inNode);
    EXPECT_EQ(nn.getOutNode(edge), outNode);
    EXPECT_EQ(nn.getInNode(newOutgoingEdge), newNode);
    EXPECT_EQ(nn.getOutNode(newOutgoingEdge), outNode);
    EXPECT_EQ(nn.getInNode(newIncomingEdge2), newNode);
    EXPECT_EQ(nn.getOutNode(newIncomingEdge2), newNode2);
    EXPECT_EQ(nn.getInNode(newOutgoingEdge2), newNode2);
    EXPECT_EQ(nn.getOutNode(newOutgoingEdge2), outNode);
    EXPECT_EQ(nn.getIncomingEdges(inNode).size(), 0);
    EXPECT_EQ(nn.getIncomingEdges(newNode).size(), 1);
    EXPECT_EQ(nn.getIncomingEdges(newNode)[0], newIncomingEdge);
    EXPECT_EQ(nn.getIncomingEdges(newNode2).size(), 1);
    EXPECT_EQ(nn.getIncomingEdges(newNode2)[0], newIncomingEdge2);
    EXPECT_EQ(nn.getIncomingEdges(outNode).size(), 3);
    EXPECT_EQ(nn.getIncomingEdges(outNode)[0], edge);
    EXPECT_EQ(nn.getIncomingEdges(outNode)[1], newOutgoingEdge);
    EXPECT_EQ(nn.getIncomingEdges(outNode)[2], newOutgoingEdge2);
    EXPECT_EQ(nn.getOutgoingEdges(inNode).size(), 2);
    EXPECT_EQ(nn.getOutgoingEdges(inNode)[0], edge);
    EXPECT_EQ(nn.getOutgoingEdges(inNode)[1], newIncomingEdge);
    EXPECT_EQ(nn.getOutgoingEdges(newNode).size(), 2);
    EXPECT_EQ(nn.getOutgoingEdges(newNode)[0], newOutgoingEdge);
    EXPECT_EQ(nn.getOutgoingEdges(newNode)[1], newIncomingEdge2);
    EXPECT_EQ(nn.getOutgoingEdges(newNode2).size(), 1);
    EXPECT_EQ(nn.getOutgoingEdges(newNode2)[0], newOutgoingEdge2);
    EXPECT_EQ(nn.getOutgoingEdges(outNode).size(), 0);
}

TEST(NeuralNetwork, AddEdge)
{
    // Set up node and edges.
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode1(2);
    NodeId outNode2(3);
    NodeId hiddenNode1(4);
    NodeId hiddenNode2(5);

    NN::Nodes nodes;
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

    NN::Edges edges;
    edges.insert({ edge1, Edge(inNode1, hiddenNode1, 0.5f) });
    edges.insert({ edge2, Edge(inNode2, hiddenNode2, 0.5f) });
    edges.insert({ edge3, Edge(hiddenNode1, outNode1, 0.5f) });
    edges.insert({ edge4, Edge(hiddenNode2, outNode2, 0.5f) });

    NN::NodeIds inputNodes;
    inputNodes.push_back(inNode1);
    inputNodes.push_back(inNode2);
    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    // Create a NeuralNetwork.
    NN nn(nodes, edges, inputNodes, outputNodes);

    EXPECT_TRUE(nn.validate());
    EXPECT_EQ(nn.getNumNodes(), 6);
    int numEdges = 4;
    EXPECT_EQ(nn.getNumEdges(), numEdges);

    // Add an edge.
    EdgeId edge5(5);
    EXPECT_TRUE(nn.addEdgeAt(inNode1, hiddenNode2, edge5, 0.1f));
    EXPECT_TRUE(nn.hasEdge(edge5));
    EXPECT_EQ(nn.getNumEdges(), ++numEdges);
    EXPECT_EQ(nn.getWeight(edge5), 0.1f);
    EXPECT_EQ(nn.getInNode(edge5), inNode1);
    EXPECT_EQ(nn.getOutNode(edge5), hiddenNode2);
    EXPECT_EQ(nn.getIncomingEdges(hiddenNode2).size(), 2);
    EXPECT_EQ(nn.getIncomingEdges(hiddenNode2)[0], edge2);
    EXPECT_EQ(nn.getIncomingEdges(hiddenNode2)[1], edge5);
    EXPECT_EQ(nn.getOutgoingEdges(inNode1).size(), 2);
    EXPECT_EQ(nn.getOutgoingEdges(inNode1)[0], edge1);
    EXPECT_EQ(nn.getOutgoingEdges(inNode1)[1], edge5);
    EXPECT_EQ(nn.getOutgoingEdges(inNode2).size(), 1);
    EXPECT_EQ(nn.getOutgoingEdges(inNode2)[0], edge2);
    EXPECT_EQ(nn.getOutgoingEdges(hiddenNode1).size(), 1);
    EXPECT_EQ(nn.getOutgoingEdges(hiddenNode1)[0], edge3);
    EXPECT_EQ(nn.getOutgoingEdges(hiddenNode2).size(), 1);
    EXPECT_EQ(nn.getOutgoingEdges(hiddenNode2)[0], edge4);

    // Try to add an edge at nodes which are already connected.
    {
        EdgeId e(6);
        EXPECT_FALSE(nn.addEdgeAt(inNode1, hiddenNode1, e, 0.5f));
        EXPECT_EQ(nn.getNumEdges(), numEdges);
        EXPECT_FALSE(nn.hasEdge(e));
    }

    // Try to add an edge going from an outputNode.
    {
        EdgeId e6(6);
        EXPECT_TRUE(nn.addEdgeAt(outNode1, inNode2, e6, 0.1f));
        EXPECT_EQ(nn.getNumEdges(), ++numEdges);
        EXPECT_TRUE(nn.hasEdge(e6));
        EXPECT_EQ(nn.getIncomingEdges(inNode2).size(), 1);
        EXPECT_EQ(nn.getOutgoingEdges(outNode1).size(), 1);
        EdgeId e7(7);
        EXPECT_TRUE(nn.addEdgeAt(outNode2, hiddenNode1, e7, 0.1f));
        EXPECT_EQ(nn.getNumEdges(), ++numEdges);
        EXPECT_TRUE(nn.hasEdge(e7));
    }

    // Add an edge going into an inputNode.
    {
        EdgeId e(8);
        EXPECT_TRUE(nn.addEdgeAt(inNode1, inNode2, e, 0.2f));
        EXPECT_TRUE(nn.hasEdge(e));
        EXPECT_EQ(nn.getNumEdges(), ++numEdges);
        EXPECT_EQ(nn.getIncomingEdges(inNode2).size(), 2);
        EXPECT_EQ(nn.getOutgoingEdges(inNode1).size(), 3);
    }

    // Try to add an edge at a node which doesn't exit.
    {
        EdgeId e(9);
        EXPECT_FALSE(nn.addEdgeAt(hiddenNode1, NodeId(6), e, 0.1f));
        EXPECT_EQ(nn.getNumEdges(), numEdges);
        EXPECT_FALSE(nn.hasEdge(e));
        EXPECT_FALSE(nn.addEdgeAt(NodeId(7), outNode1, e, 0.1f));
        EXPECT_EQ(nn.getNumEdges(), numEdges);
        EXPECT_FALSE(nn.hasEdge(e));
    }

    // Try to add an edge which creates a circle.
    {
        EdgeId e(9);
        EXPECT_TRUE(nn.addEdgeAt(hiddenNode2, inNode1, e, 0.1f));
        EXPECT_EQ(nn.getNumEdges(), ++numEdges);
        EXPECT_TRUE(nn.hasEdge(e));
        EXPECT_EQ(nn.getIncomingEdges(inNode1).size(), 1);
        EXPECT_EQ(nn.getOutgoingEdges(hiddenNode2).size(), 2);
    }
}

TEST(NeuralNetwork, ReplaceEdge)
{
    // Set up node and edges.
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode1(2);
    NodeId outNode2(3);
    NodeId hiddenNode1(4);
    NodeId hiddenNode2(5);

    NN::Nodes nodes;
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

    NN::Edges edges;
    edges.insert({ edge1, Edge(inNode1, hiddenNode1, 0.5f) });
    edges.insert({ edge2, Edge(inNode2, hiddenNode2, 0.5f) });
    edges.insert({ edge3, Edge(hiddenNode1, outNode1, 0.5f) });
    edges.insert({ edge4, Edge(hiddenNode2, outNode2, 0.5f) });

    NN::NodeIds inputNodes;
    inputNodes.push_back(inNode1);
    inputNodes.push_back(inNode2);
    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    // Create a NeuralNetwork.
    NN nn(nodes, edges, inputNodes, outputNodes);

    EXPECT_TRUE(nn.validate());
    EXPECT_EQ(nn.getNumNodes(), 6);
    int numEdges = 4;
    EXPECT_EQ(nn.getNumEdges(), numEdges);

    // Replace an edge.
    EdgeId edge5(5);
    nn.replaceEdgeId(edge1, edge5);
    EXPECT_TRUE(nn.validate());
    EXPECT_FALSE(nn.hasEdge(edge1));
    EXPECT_TRUE(nn.hasEdge(edge5));
    EXPECT_EQ(nn.getNumEdges(), numEdges);
    EXPECT_EQ(nn.getIncomingEdges(hiddenNode1).size(), 1);
    EXPECT_EQ(nn.getIncomingEdges(hiddenNode1)[0], edge5);
    EXPECT_EQ(nn.getOutgoingEdges(inNode1).size(), 1);
    EXPECT_EQ(nn.getOutgoingEdges(inNode1)[0], edge5);
}

TEST(NeuralNetwork, RemoveEdge)
{
    // Set up node and edges.
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode1(2);
    NodeId outNode2(3);
    NodeId hiddenNode1(4);
    NodeId hiddenNode2(5);

    NN::Nodes nodes;
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

    NN::Edges edges;
    edges.insert({ edge1, Edge(inNode1, hiddenNode1, 0.5f) });
    edges.insert({ edge2, Edge(inNode2, hiddenNode2, 0.5f) });
    edges.insert({ edge3, Edge(hiddenNode1, outNode1, 0.5f) });
    edges.insert({ edge4, Edge(hiddenNode2, outNode2, 0.5f) });

    NN::NodeIds inputNodes;
    inputNodes.push_back(inNode1);
    inputNodes.push_back(inNode2);
    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    // Create a NeuralNetwork.
    NN nn(nodes, edges, inputNodes, outputNodes);

    EXPECT_TRUE(nn.validate());
    EXPECT_EQ(nn.getNumNodes(), 6);
    int numEdges = 4;
    EXPECT_EQ(nn.getNumEdges(), numEdges);

    // Remove an edge.
    nn.removeEdge(edge1);
    EXPECT_TRUE(nn.validate());
    EXPECT_FALSE(nn.hasEdge(edge1));
    EXPECT_EQ(nn.getNumEdges(), numEdges - 1);
    EXPECT_EQ(nn.getIncomingEdges(hiddenNode1).size(), 0);
    EXPECT_EQ(nn.getOutgoingEdges(inNode1).size(), 0);
}

TEST(NeuralNetwork, ReplaceNode)
{
    // Set up node and edges.
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode1(2);
    NodeId outNode2(3);
    NodeId hiddenNode1(4);
    NodeId hiddenNode2(5);

    NN::Nodes nodes;
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

    NN::Edges edges;
    edges.insert({ edge1, Edge(inNode1, hiddenNode1, 0.5f) });
    edges.insert({ edge2, Edge(inNode2, hiddenNode2, 0.5f) });
    edges.insert({ edge3, Edge(hiddenNode1, outNode1, 0.5f) });
    edges.insert({ edge4, Edge(hiddenNode2, outNode2, 0.5f) });

    NN::NodeIds inputNodes;
    inputNodes.push_back(inNode1);
    inputNodes.push_back(inNode2);
    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    // Create a NeuralNetwork.
    NN nn(nodes, edges, inputNodes, outputNodes);

    EXPECT_TRUE(nn.validate());
    EXPECT_EQ(nn.getNumNodes(), 6);
    int numEdges = 4;
    EXPECT_EQ(nn.getNumEdges(), numEdges);

    // Replace a node.
    NodeId newNode(6);
    nn.replaceNodeId(outNode1, newNode);
    EXPECT_TRUE(nn.validate());
    EXPECT_TRUE(nn.hasNode(newNode));
    EXPECT_FALSE(nn.hasNode(outNode1));
    EXPECT_EQ(nn.getNumNodes(), 6);
    EXPECT_EQ(nn.getNumEdges(), numEdges);
    EXPECT_EQ(nn.getIncomingEdges(newNode).size(), 1);
    EXPECT_EQ(nn.getIncomingEdges(newNode)[0], edge3);
    EXPECT_EQ(nn.getOutgoingEdges(newNode).size(), 0);
    EXPECT_EQ(nn.getOutNode(edge3), newNode);
    EXPECT_EQ(nn.getOutputNodes().size(), 2);
    EXPECT_TRUE(nn.getOutputNodes()[0] == newNode || nn.getOutputNodes()[1] == newNode);
}

TEST(NeuralNetwork, Evaluate)
{
    // Create a NN looks like below

    // 5.0 (0) -1.0-> (2) -(-3.0)-> (4)
    //                              /
    // 6.0 (1) -2.0-> (3) --- 4.0 -/

    NodeId n0(0);
    NodeId n1(1);
    NodeId n2(2);
    NodeId n3(3);
    NodeId n4(4);

    EdgeId e0(0);
    EdgeId e1(1);
    EdgeId e2(2);
    EdgeId e3(3);

    NN::Nodes nodes;
    nodes.insert({ n0, Node(5.0f) });
    nodes.insert({ n1, Node(6.0f) });
    nodes.insert({ n2, Node(0.0f) });
    nodes.insert({ n3, Node(0.0f) });
    nodes.insert({ n4, Node(0.0f) });

    NN::Edges edges;
    edges.insert({ e0, Edge(n0, n2, 1.0f) });
    edges.insert({ e1, Edge(n1, n3, 2.0f) });
    edges.insert({ e2, Edge(n2, n4, -3.0f) });
    edges.insert({ e3, Edge(n3, n4, 4.0f) });

    NN::NodeIds inputNodes;
    inputNodes.push_back(n0);
    inputNodes.push_back(n1);
    NN::NodeIds outputNodes;
    outputNodes.push_back(n4);

    // Create a NeuralNetwork.
    NN nn(nodes, edges, inputNodes, outputNodes);

    // Evaluate
    nn.evaluate();

    EXPECT_EQ(nn.getNode(n4).getValue(), 33.f); // -3 * (5 * 1) + 4 * (6 * 2) = 33.0f

    // Evaluating multiple times shouldn't change the result for feed forward network.
    nn.evaluate();

    EXPECT_EQ(nn.getNode(n4).getValue(), 33.f);
}

TEST(NeuralNetwork, EvaluateRecurrent)
{
    // Create a NN looks like below

    //                _9.0
    //                \ /
    // 5.0 (0) -1.0-> (2) -(-3.0)-> (4)
    //
    // 6.0 (1) -2.0-> (3) -4.0-> (5) -7.0-> (6)
    //                 |____8.0___|

    NodeId n0(0);
    NodeId n1(1);
    NodeId n2(2);
    NodeId n3(3);
    NodeId n4(4);
    NodeId n5(5);
    NodeId n6(6);

    EdgeId e0(0);
    EdgeId e1(1);
    EdgeId e2(2);
    EdgeId e3(3);
    EdgeId e4(4);
    EdgeId e5(5);
    EdgeId e6(6);

    NN::Nodes nodes;
    nodes.insert({ n0, Node(5.0f) });
    nodes.insert({ n1, Node(6.0f) });
    nodes.insert({ n2, Node(0.0f) });
    nodes.insert({ n3, Node(0.0f) });
    nodes.insert({ n4, Node(0.0f) });
    nodes.insert({ n5, Node(0.0f) });
    nodes.insert({ n6, Node(0.0f) });

    NN::Edges edges;
    edges.insert({ e0, Edge(n0, n2, 1.0f) });
    edges.insert({ e1, Edge(n2, n2, 9.0f) });
    edges.insert({ e2, Edge(n2, n4, -3.0f) });
    edges.insert({ e3, Edge(n1, n3, 2.0f) });
    edges.insert({ e4, Edge(n3, n5, 4.0f) });
    edges.insert({ e5, Edge(n5, n3, 8.0f) });
    edges.insert({ e6, Edge(n5, n6, 7.0f) });

    NN::NodeIds inputNodes;
    inputNodes.push_back(n0);
    inputNodes.push_back(n1);
    NN::NodeIds outputNodes;
    outputNodes.push_back(n4);
    outputNodes.push_back(n6);

    // Create a NeuralNetwork.
    NN nn(nodes, edges, inputNodes, outputNodes);

    // Evaluate
    nn.evaluate();

    EXPECT_EQ(nn.getNode(n4).getValue(), -15.f); // -3 * (5 * 1) = -15.0f
    EXPECT_EQ(nn.getNode(n6).getValue(), 336.f); // 7 * (4 * (6 * 2)) = 336.f;
    EXPECT_EQ(nn.getNode(n2).getValue(), 5.f); // 5 * 1 = 5.0f
    EXPECT_EQ(nn.getNode(n5).getValue(), 48.f); // 4 * (6 * 2) = 336.f;

    // Evaluate again
    nn.evaluate();
    EXPECT_EQ(nn.getNode(n4).getValue(), -150.f); // -3 * (5 * 1 + 9 * 5) = -150.0f
    EXPECT_EQ(nn.getNode(n6).getValue(), 11088.f); // 7 * (4 * (6 * 2 + 48 * 8)) = 11088.f;
}