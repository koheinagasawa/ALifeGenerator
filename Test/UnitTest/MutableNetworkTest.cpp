/*
* MutableNetworkTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/MutableNetwork.h>

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
    nodes[inNode] = Node();
    nodes[outNode] = Node();

    EdgeId edge(0);

    MN::Edges edges;
    edges[edge] = MN::Edge(inNode, outNode, 0.5f);

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
    nodes[inNode] = Node();
    nodes[outNode] = Node();

    EdgeId edge(0);

    MN::Edges edges;
    edges[edge] = MN::Edge(inNode, outNode, 0.5f);

    MN::NodeIds outputNodes;
    outputNodes.push_back(outNode);

    MN mn(nodes, edges, outputNodes);

    EXPECT_TRUE(mn.validate());
    EXPECT_EQ(mn.getNumNodes(), 2);
    EXPECT_EQ(mn.getNumEdges(), 1);
    EXPECT_TRUE(mn.isEdgeEnabled(edge));
    EXPECT_EQ(mn.getWeight(edge), 0.5f);

    // Try to add a node at an edge which doesn't exist.
    NodeId newNode;
    EdgeId newIncomingEdge;
    EdgeId newOutgoingEdge;
    mn.addNodeAt(EdgeId(1), newNode, newIncomingEdge, newOutgoingEdge);
    EXPECT_EQ(mn.getNumNodes(), 2);
    EXPECT_EQ(mn.getNumEdges(), 1);
    EXPECT_FALSE(newNode.isValid());
    EXPECT_FALSE(newIncomingEdge.isValid());
    EXPECT_FALSE(newOutgoingEdge.isValid());

    // Add one node
    mn.addNodeAt(edge, newNode, newIncomingEdge, newOutgoingEdge);

    EXPECT_TRUE(newNode != inNode);
    EXPECT_TRUE(newNode != outNode);
    EXPECT_TRUE(newIncomingEdge != edge);
    EXPECT_TRUE(newOutgoingEdge != edge);
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
    NodeId newNode2;
    EdgeId newIncomingEdge2, newOutgoingEdge2;
    mn.addNodeAt(newOutgoingEdge, newNode2, newIncomingEdge2, newOutgoingEdge2);

    EXPECT_TRUE(newNode != inNode);
    EXPECT_TRUE(newNode != outNode);
    EXPECT_TRUE(newOutgoingEdge != edge);
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
    nodes[inNode1] = Node();
    nodes[inNode2] = Node();
    nodes[outNode1] = Node();
    nodes[outNode2] = Node();
    nodes[hiddenNode1] = Node();
    nodes[hiddenNode2] = Node();

    EdgeId edge1(0);
    EdgeId edge2(1);
    EdgeId edge3(2);
    EdgeId edge4(3);

    MN::Edges edges;
    edges[edge1] = MN::Edge(inNode1, hiddenNode1, 0.5f);
    edges[edge2] = MN::Edge(inNode2, hiddenNode2, 0.5f);
    edges[edge3] = MN::Edge(hiddenNode1, outNode1, 0.5f);
    edges[edge4] = MN::Edge(hiddenNode2, outNode2, 0.5f);

    MN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    MN mn(nodes, edges, outputNodes);

    EXPECT_TRUE(mn.validate());
    EXPECT_EQ(mn.getNumNodes(), 6);
    int numEdges = 4;
    EXPECT_EQ(mn.getNumEdges(), numEdges);

    // Add an edge.
    EdgeId edge5 = mn.addEdgeAt(inNode1, hiddenNode2, 0.1f);
    EXPECT_TRUE(edge5.isValid());
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
        EdgeId e = mn.addEdgeAt(inNode1, hiddenNode1, 0.5f);
        EXPECT_FALSE(e.isValid());
        EXPECT_EQ(mn.getNumEdges(), numEdges);
    }

    // Try to add an edge going from an outputNode.
    {
        EdgeId e = mn.addEdgeAt(outNode1, inNode2, 0.1f);
        EXPECT_FALSE(e.isValid());
        EXPECT_EQ(mn.getNumEdges(), numEdges);
        e = mn.addEdgeAt(outNode2, hiddenNode1, 0.1f);
        EXPECT_FALSE(e.isValid());
        EXPECT_EQ(mn.getNumEdges(), numEdges);
    }

    // Add an edge going into an inputNode.
    // This is fine and shouldn't fail because we don't differentiate input nodes and hidden nodes internally.
    EdgeId edge6 = mn.addEdgeAt(inNode1, inNode2, 0.2f);
    EXPECT_TRUE(edge6.isValid());
    EXPECT_TRUE(mn.hasEdge(edge6));
    EXPECT_EQ(mn.getNumEdges(), ++numEdges);
    EXPECT_EQ(mn.getWeight(edge6), 0.2f);
    EXPECT_EQ(mn.getInNode(edge6), inNode1);
    EXPECT_EQ(mn.getOutNode(edge6), inNode2);
    EXPECT_EQ(mn.getIncomingEdges(inNode2).size(), 1);
    EXPECT_EQ(mn.getIncomingEdges(inNode2)[0], edge6);

    // Try to add an edge at a node which doesn't exit.
    {
        EdgeId e = mn.addEdgeAt(hiddenNode1, NodeId(6), 0.1f);
        EXPECT_FALSE(e.isValid());
        EXPECT_EQ(mn.getNumEdges(), numEdges);
        e = mn.addEdgeAt(NodeId(7), outNode1, 0.1f);
        EXPECT_FALSE(e.isValid());
        EXPECT_EQ(mn.getNumEdges(), numEdges);
    }

    // Try to add an edge which creates a circle.
    {
        EdgeId e = mn.addEdgeAt(hiddenNode2, inNode1, 0.1f);
        EXPECT_FALSE(e.isValid());
        EXPECT_EQ(mn.getNumEdges(), numEdges);
        EXPECT_EQ(mn.getIncomingEdges(inNode1).size(), 0);
    }
}
