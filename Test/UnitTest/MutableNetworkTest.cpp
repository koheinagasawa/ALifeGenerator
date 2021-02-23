/*
* MutableNetworkTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/NeuralNetwork/MutableNetwork.h>

// Basic node class.
struct Node : public NodeBase
{
    Node() = default;
    Node(float value) : m_value(value) {}

    virtual float getValue() const { return m_value; }
    virtual void setValue(float value) { m_value = value; }

    float m_value = 0.f;
};

using MN = MutableNetwork<Node>;

TEST(MutableNetwork, EnableDisableEdge)
{
    NodeId inNode(0);
    NodeId outNode(1);

    MN::Nodes nodes;
    nodes.insert({ inNode, Node() });
    nodes.insert({ outNode, Node() });

    EdgeId edge(0);

    MN::Edges edges;
    edges.insert({ edge, MN::Edge(inNode, outNode, 0.5f) });

    MN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    MN mn(nodes, edges, outputNodes);

    EXPECT_TRUE(mn.validate());
    EXPECT_EQ(mn.getNumNodes(), 2);
    EXPECT_EQ(mn.getNumEdges(), 1);
    EXPECT_TRUE(mn.isEdgeEnabled(edge));
    EXPECT_EQ(mn.getWeight(edge), 0.5f);

    mn.setEdgeEnabled(edge, false);
    EXPECT_FALSE(mn.isEdgeEnabled(edge));
    EXPECT_EQ(mn.getWeight(edge), 0);
    EXPECT_EQ(mn.getWeightRaw(edge), 0.5f);

    mn.setEdgeEnabled(edge, true);
    EXPECT_TRUE(mn.isEdgeEnabled(edge));
    EXPECT_EQ(mn.getWeight(edge), 0.5f);
    EXPECT_EQ(mn.getWeightRaw(edge), 0.5f);
}

TEST(MutableNetwork, AddNode)
{
    NodeId inNode(0);
    NodeId outNode(1);

    MN::Nodes nodes;
    nodes.insert({ inNode, Node() });
    nodes.insert({ outNode, Node() });

    EdgeId edge(0);

    MN::Edges edges;
    edges.insert({ edge, MN::Edge(inNode, outNode, 0.5f) });

    MN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    MN mn(nodes, edges, outputNodes);

    EXPECT_TRUE(mn.validate());
    EXPECT_EQ(mn.getNumNodes(), 2);
    EXPECT_EQ(mn.getNumEdges(), 1);
    EXPECT_TRUE(mn.isEdgeEnabled(edge));
    EXPECT_EQ(mn.getWeight(edge), 0.5f);

    // Try to add a node at an edge which doesn't exist.
    NodeId newNode(2);
    EdgeId newIncomingEdge(1);
    EdgeId newOutgoingEdge(2);
    mn.addNodeAt(EdgeId(1), newNode, newIncomingEdge, newOutgoingEdge);
    EXPECT_EQ(mn.getNumNodes(), 2);
    EXPECT_EQ(mn.getNumEdges(), 1);

    // Add one node
    mn.addNodeAt(edge, newNode, newIncomingEdge, newOutgoingEdge);

    EXPECT_TRUE(mn.hasNode(inNode));
    EXPECT_TRUE(mn.hasNode(outNode));
    EXPECT_TRUE(mn.hasNode(newNode));
    EXPECT_TRUE(mn.hasEdge(edge));
    EXPECT_TRUE(mn.hasEdge(newIncomingEdge));
    EXPECT_TRUE(mn.hasEdge(newOutgoingEdge));
    EXPECT_FALSE(mn.isEdgeEnabled(edge));
    EXPECT_TRUE(mn.isEdgeEnabled(newIncomingEdge));
    EXPECT_TRUE(mn.isEdgeEnabled(newOutgoingEdge));
    EXPECT_EQ(mn.getWeight(newIncomingEdge), 1.0f);
    EXPECT_EQ(mn.getWeight(newOutgoingEdge), 0.5f);
    EXPECT_EQ(mn.getNumNodes(), 3);
    EXPECT_EQ(mn.getNumEdges(), 3);
    EXPECT_EQ(mn.getInNode(edge), inNode);
    EXPECT_EQ(mn.getOutNode(edge), outNode);
    EXPECT_EQ(mn.getInNode(newIncomingEdge), inNode);
    EXPECT_EQ(mn.getOutNode(newIncomingEdge), newNode);
    EXPECT_EQ(mn.getInNode(newOutgoingEdge), newNode);
    EXPECT_EQ(mn.getOutNode(newOutgoingEdge), outNode);
    EXPECT_EQ(mn.getIncomingEdges(inNode).size(), 0);
    EXPECT_EQ(mn.getIncomingEdges(newNode).size(), 1);
    EXPECT_EQ(mn.getIncomingEdges(newNode)[0], newIncomingEdge);
    EXPECT_EQ(mn.getIncomingEdges(outNode).size(), 2);
    EXPECT_EQ(mn.getIncomingEdges(outNode)[0], edge);
    EXPECT_EQ(mn.getIncomingEdges(outNode)[1], newOutgoingEdge);

    // Add one more node
    NodeId newNode2(3);
    EdgeId newIncomingEdge2(3), newOutgoingEdge2(4);
    mn.addNodeAt(newOutgoingEdge, newNode2, newIncomingEdge2, newOutgoingEdge2);

    EXPECT_TRUE(mn.hasNode(inNode));
    EXPECT_TRUE(mn.hasNode(outNode));
    EXPECT_TRUE(mn.hasNode(newNode));
    EXPECT_TRUE(mn.hasNode(newNode2));
    EXPECT_TRUE(mn.hasEdge(edge));
    EXPECT_TRUE(mn.hasEdge(newOutgoingEdge));
    EXPECT_TRUE(mn.hasEdge(newIncomingEdge2));
    EXPECT_TRUE(mn.hasEdge(newOutgoingEdge2));
    EXPECT_FALSE(mn.isEdgeEnabled(edge));
    EXPECT_TRUE(mn.isEdgeEnabled(newIncomingEdge));
    EXPECT_FALSE(mn.isEdgeEnabled(newOutgoingEdge));
    EXPECT_TRUE(mn.isEdgeEnabled(newIncomingEdge2));
    EXPECT_TRUE(mn.isEdgeEnabled(newOutgoingEdge2));
    EXPECT_EQ(mn.getWeight(newIncomingEdge), 1.f);
    EXPECT_EQ(mn.getWeight(newIncomingEdge2), 1.f);
    EXPECT_EQ(mn.getWeight(newOutgoingEdge2), 0.5f);
    EXPECT_EQ(mn.getNumNodes(), 4);
    EXPECT_EQ(mn.getNumEdges(), 5);
    EXPECT_EQ(mn.getInNode(edge), inNode);
    EXPECT_EQ(mn.getOutNode(edge), outNode);
    EXPECT_EQ(mn.getInNode(newOutgoingEdge), newNode);
    EXPECT_EQ(mn.getOutNode(newOutgoingEdge), outNode);
    EXPECT_EQ(mn.getInNode(newIncomingEdge2), newNode);
    EXPECT_EQ(mn.getOutNode(newIncomingEdge2), newNode2);
    EXPECT_EQ(mn.getInNode(newOutgoingEdge2), newNode2);
    EXPECT_EQ(mn.getOutNode(newOutgoingEdge2), outNode);
    EXPECT_EQ(mn.getIncomingEdges(inNode).size(), 0);
    EXPECT_EQ(mn.getIncomingEdges(newNode).size(), 1);
    EXPECT_EQ(mn.getIncomingEdges(newNode)[0], newIncomingEdge);
    EXPECT_EQ(mn.getIncomingEdges(newNode2).size(), 1);
    EXPECT_EQ(mn.getIncomingEdges(newNode2)[0], newIncomingEdge2);
    EXPECT_EQ(mn.getIncomingEdges(outNode).size(), 3);
    EXPECT_EQ(mn.getIncomingEdges(outNode)[0], edge);
    EXPECT_EQ(mn.getIncomingEdges(outNode)[1], newOutgoingEdge);
    EXPECT_EQ(mn.getIncomingEdges(outNode)[2], newOutgoingEdge2);
}

TEST(MutableNetwork, AddEdge)
{
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode1(2);
    NodeId outNode2(3);
    NodeId hiddenNode1(4);
    NodeId hiddenNode2(5);

    MN::Nodes nodes;
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

    MN::Edges edges;
    edges.insert({ edge1, MN::Edge(inNode1, hiddenNode1, 0.5f) });
    edges.insert({ edge2, MN::Edge(inNode2, hiddenNode2, 0.5f) });
    edges.insert({ edge3, MN::Edge(hiddenNode1, outNode1, 0.5f) });
    edges.insert({ edge4, MN::Edge(hiddenNode2, outNode2, 0.5f) });

    MN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    MN mn(nodes, edges, outputNodes);

    EXPECT_TRUE(mn.validate());
    EXPECT_EQ(mn.getNumNodes(), 6);
    int numEdges = 4;
    EXPECT_EQ(mn.getNumEdges(), numEdges);

    // Add an edge.
    EdgeId edge5(5);
    EXPECT_TRUE(mn.addEdgeAt(inNode1, hiddenNode2, edge5, 0.1f));
    EXPECT_TRUE(mn.hasEdge(edge5));
    EXPECT_EQ(mn.getNumEdges(), ++numEdges);
    EXPECT_EQ(mn.getWeight(edge5), 0.1f);
    EXPECT_EQ(mn.getInNode(edge5), inNode1);
    EXPECT_EQ(mn.getOutNode(edge5), hiddenNode2);
    EXPECT_EQ(mn.getIncomingEdges(hiddenNode2).size(), 2);
    EXPECT_EQ(mn.getIncomingEdges(hiddenNode2)[0], edge2);
    EXPECT_EQ(mn.getIncomingEdges(hiddenNode2)[1], edge5);

    // Try to add an edge at nodes which are already connected.
    {
        EdgeId e(6);
        EXPECT_FALSE(mn.addEdgeAt(inNode1, hiddenNode1, e, 0.5f));
        EXPECT_EQ(mn.getNumEdges(), numEdges);
        EXPECT_FALSE(mn.hasEdge(e));
    }

    // Try to add an edge going from an outputNode.
    {
        EdgeId e(6);
        EXPECT_FALSE(mn.addEdgeAt(outNode1, inNode2, e, 0.1f));
        EXPECT_EQ(mn.getNumEdges(), numEdges);
        EXPECT_FALSE(mn.hasEdge(e));
        EXPECT_FALSE(mn.addEdgeAt(outNode2, hiddenNode1, e, 0.1f));
        EXPECT_EQ(mn.getNumEdges(), numEdges);
        EXPECT_FALSE(mn.hasEdge(e));
    }

    // Add an edge going into an inputNode.
    // This is fine and shouldn't fail because we don't differentiate input nodes and hidden nodes internally.
    EdgeId edge6(6);
    EXPECT_TRUE(mn.addEdgeAt(inNode1, inNode2, edge6, 0.2f));
    EXPECT_TRUE(mn.hasEdge(edge6));
    EXPECT_EQ(mn.getNumEdges(), ++numEdges);
    EXPECT_EQ(mn.getWeight(edge6), 0.2f);
    EXPECT_EQ(mn.getInNode(edge6), inNode1);
    EXPECT_EQ(mn.getOutNode(edge6), inNode2);
    EXPECT_EQ(mn.getIncomingEdges(inNode2).size(), 1);
    EXPECT_EQ(mn.getIncomingEdges(inNode2)[0], edge6);

    // Try to add an edge at a node which doesn't exit.
    {
        EdgeId e(7);
        EXPECT_FALSE(mn.addEdgeAt(hiddenNode1, NodeId(6), e, 0.1f));
        EXPECT_EQ(mn.getNumEdges(), numEdges);
        EXPECT_FALSE(mn.hasEdge(e));
        EXPECT_FALSE(mn.addEdgeAt(NodeId(7), outNode1, e, 0.1f));
        EXPECT_EQ(mn.getNumEdges(), numEdges);
        EXPECT_FALSE(mn.hasEdge(e));
    }

    // Try to add an edge which creates a circle.
    {
        EdgeId e(7);
        EXPECT_FALSE(mn.addEdgeAt(hiddenNode2, inNode1, e, 0.1f));
        EXPECT_EQ(mn.getNumEdges(), numEdges);
        EXPECT_FALSE(mn.hasEdge(e));
        EXPECT_EQ(mn.getIncomingEdges(inNode1).size(), 0);
    }
}

TEST(MutableNetwork, ReplaceEdge)
{
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode1(2);
    NodeId outNode2(3);
    NodeId hiddenNode1(4);
    NodeId hiddenNode2(5);

    MN::Nodes nodes;
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

    MN::Edges edges;
    edges.insert({ edge1, MN::Edge(inNode1, hiddenNode1, 0.5f) });
    edges.insert({ edge2, MN::Edge(inNode2, hiddenNode2, 0.5f) });
    edges.insert({ edge3, MN::Edge(hiddenNode1, outNode1, 0.5f) });
    edges.insert({ edge4, MN::Edge(hiddenNode2, outNode2, 0.5f) });

    MN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    MN mn(nodes, edges, outputNodes);

    EXPECT_TRUE(mn.validate());
    EXPECT_EQ(mn.getNumNodes(), 6);
    int numEdges = 4;
    EXPECT_EQ(mn.getNumEdges(), numEdges);

    // Remove an edge.
    EdgeId edge5(5);
    mn.replaceEdgeId(edge1, edge5);
    EXPECT_TRUE(mn.validate());
    EXPECT_FALSE(mn.hasEdge(edge1));
    EXPECT_TRUE(mn.hasEdge(edge5));
    EXPECT_EQ(mn.getNumEdges(), numEdges);
    EXPECT_EQ(mn.getIncomingEdges(hiddenNode1).size(), 1);
    EXPECT_EQ(mn.getIncomingEdges(hiddenNode1)[0], edge5);
}

TEST(MutableNetwork, ReplaceNode)
{
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode1(2);
    NodeId outNode2(3);
    NodeId hiddenNode1(4);
    NodeId hiddenNode2(5);

    MN::Nodes nodes;
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

    MN::Edges edges;
    edges.insert({ edge1, MN::Edge(inNode1, hiddenNode1, 0.5f) });
    edges.insert({ edge2, MN::Edge(inNode2, hiddenNode2, 0.5f) });
    edges.insert({ edge3, MN::Edge(hiddenNode1, outNode1, 0.5f) });
    edges.insert({ edge4, MN::Edge(hiddenNode2, outNode2, 0.5f) });

    MN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    MN mn(nodes, edges, outputNodes);

    EXPECT_TRUE(mn.validate());
    EXPECT_EQ(mn.getNumNodes(), 6);
    int numEdges = 4;
    EXPECT_EQ(mn.getNumEdges(), numEdges);

    NodeId newNode(6);
    mn.replaceNodeId(outNode1, newNode);
    EXPECT_TRUE(mn.validate());
    EXPECT_TRUE(mn.hasNode(newNode));
    EXPECT_FALSE(mn.hasNode(outNode1));
    EXPECT_EQ(mn.getNumNodes(), 6);
    EXPECT_EQ(mn.getNumEdges(), numEdges);
    EXPECT_EQ(mn.getIncomingEdges(newNode).size(), 1);
    EXPECT_EQ(mn.getIncomingEdges(newNode)[0], edge3);
    EXPECT_EQ(mn.getOutNode(edge3), newNode);
    EXPECT_EQ(mn.getNumOutputNodes(), 2);
    EXPECT_TRUE(mn.getOutputNodes()[0] == newNode || mn.getOutputNodes()[1] == newNode);
}
