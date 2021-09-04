/*
* NeuralNetworkEvalatorTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>
#include <UnitTest/UnitTestBaseTypes.h>

#include <NEAT/NeuralNetwork/NeuralNetworkEvaluator.h>

using NN = NeuralNetwork<Node, Edge>;

TEST(NeuralNetworkEvaluator, Evaluate)
{
    // Create a NN looks like below

    //                _0.2
    //                \ /
    // 5.0 (0) -1.0-> (2) -(-3.0)-> (4)
    //
    // 6.0 (1) -2.0-> (3) -0.1-> (5) -7.0-> (6)
    //                 |____0.3___|

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
    edges.insert({ e1, Edge(n2, n2, 0.2f) });
    edges.insert({ e2, Edge(n2, n4, -3.0f) });
    edges.insert({ e3, Edge(n1, n3, 2.0f) });
    edges.insert({ e4, Edge(n3, n5, 0.1f) });
    edges.insert({ e5, Edge(n5, n3, 0.3f) });
    edges.insert({ e6, Edge(n5, n6, 7.0f) });

    NN::NodeIds inputNodes;
    inputNodes.push_back(n0);
    inputNodes.push_back(n1);
    NN::NodeIds outputNodes;
    outputNodes.push_back(n4);
    outputNodes.push_back(n6);

    // Create a NeuralNetwork.
    NN nn(nodes, edges, inputNodes, outputNodes);

    NeuralNetworkEvaluator evaluator;
    evaluator.m_type = NeuralNetworkEvaluator::EvaluationType::ITERATION;
    evaluator.m_evalIterations = 2;

    evaluator.evaluate(nn.getOutputNodes(), &nn);
    EXPECT_TRUE(std::fabs(nn.getNode(n4).getValue() - (-18.f)) < 1e-4f); // -3 * (5 * 1 + 0.2 * 5) = -18.0f
    EXPECT_TRUE(std::fabs(nn.getNode(n6).getValue() - 8.652f) < 1e-4f); // 7 * (0.1 * (6 * 2 + 1.2 * 0.3)) = 8.652f;

    nn.setNodeValue(n2, 0.f);
    nn.setNodeValue(n3, 0.f);
    nn.setNodeValue(n4, 0.f);
    nn.setNodeValue(n5, 0.f);
    nn.setNodeValue(n6, 0.f);

    evaluator.m_type = NeuralNetworkEvaluator::EvaluationType::CONVERGE;
    evaluator.m_convergenceThreshold = 1e-6f;
    evaluator.m_evalIterations = 10000;

    evaluator.evaluate(nn.getOutputNodes(), &nn);
    EXPECT_TRUE(std::fabs(nn.getNode(n4).getValue() - (-18.75f)) < 1e-4f); // -3 * (5 * 1 + 5 / 4) = -18.75f
    EXPECT_TRUE(evaluator.getCurrentIteration() < evaluator.m_evalIterations);
}
