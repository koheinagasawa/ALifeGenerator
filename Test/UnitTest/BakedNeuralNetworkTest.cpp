/*
* NeuralNetworkTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>
#include <UnitTest/UnitTestBaseTypes.h>

#include <EvoAlgo/NeuralNetwork/Node.h>
#include <EvoAlgo/NeuralNetwork/Edge.h>
#include <EvoAlgo/NeuralNetwork/NeuralNetwork.h>
#include <EvoAlgo/NeuralNetwork/BakedNeuralNetwork.h>

using NN = NeuralNetwork<DefaultNode, DefaultEdge>;

TEST(BakedNeuralNetwork, CompareEvalResult)
{
    // Set up node and edges.
    NodeId inNode1(0);
    NodeId inNode2(1);
    NodeId outNode1(2);
    NodeId outNode2(3);
    NodeId hiddenNode1(4);
    NodeId hiddenNode2(5);

    NN::Nodes nodes;
    nodes.insert({ inNode1, DefaultNode() });
    nodes.insert({ inNode2, DefaultNode() });
    nodes.insert({ outNode1, DefaultNode() });
    nodes.insert({ outNode2, DefaultNode() });
    nodes.insert({ hiddenNode1, DefaultNode() });
    nodes.insert({ hiddenNode2, DefaultNode() });

    EdgeId edge1(1);
    EdgeId edge2(2);
    EdgeId edge3(3);
    EdgeId edge4(4);
    EdgeId edge5(5);
    EdgeId edge6(6);
    EdgeId edge7(7);
    EdgeId edge8(8);

    NN::Edges edges;
    edges.insert({ edge1, DefaultEdge(inNode1, hiddenNode1, 0.1f) });
    edges.insert({ edge2, DefaultEdge(inNode1, hiddenNode2, 0.2f) });
    edges.insert({ edge3, DefaultEdge(inNode2, hiddenNode1, 0.3f) });
    edges.insert({ edge4, DefaultEdge(inNode2, hiddenNode2, 0.4f) });
    edges.insert({ edge5, DefaultEdge(hiddenNode1, outNode1, 0.5f) });
    edges.insert({ edge6, DefaultEdge(hiddenNode1, outNode2, 0.6f) });
    edges.insert({ edge7, DefaultEdge(hiddenNode2, outNode1, 0.7f, false) });
    edges.insert({ edge8, DefaultEdge(hiddenNode2, outNode2, 0.8f) });

    NN::NodeIds inputNodes;
    inputNodes.push_back(inNode1);
    inputNodes.push_back(inNode2);
    NN::NodeIds outputNodes;
    outputNodes.push_back(outNode1);
    outputNodes.push_back(outNode2);

    // Create a NeuralNetwork.
    NN nn(nodes, edges, inputNodes, outputNodes);

    Activation activation1([](float value) { return 2.f * value; });
    Activation activation2([](float value) { return value + 1.f; });
    nn.accessNode(hiddenNode1).setActivation(&activation1);
    nn.accessNode(hiddenNode2).setActivation(&activation2);

    nn.setAllNodeValues(0.f);
    nn.setNodeValue(inNode1, 1.0f);
    nn.setNodeValue(inNode2, 2.0f);

    std::shared_ptr<BakedNeuralNetwork> baked = nn.bake();

    nn.evaluate();
    baked->evaluate();

    EXPECT_EQ(nn.getNode(outNode1).getValue(), baked->getNodeValue(outNode1));
    EXPECT_EQ(nn.getNode(outNode2).getValue(), baked->getNodeValue(outNode2));
}
