/*
* FeedForwardNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/NeuralNetwork.h>
#include <NEAT/NeuralNetwork/FeedForwardNetwork.h>

// Factory class of NeuralNetwork
class NeuralNetworkFactory
{
public:
    template <typename Node, typename Edge>
    static auto createNeuralNetwork(
        NeuralNetworkType type,
        const typename NeuralNetwork<Node, Edge>::Nodes& nodes,
        const typename NeuralNetwork<Node, Edge>::Edges& edges,
        const typename NeuralNetwork<Node, Edge>::NodeIds& inputNodes,
        const typename NeuralNetwork<Node, Edge>::NodeIds& outputNodes)->std::shared_ptr<NeuralNetwork<Node, Edge>>
    {
        switch (type)
        {
        case NeuralNetworkType::FEED_FORWARD:
            return std::make_shared<FeedForwardNetwork<Node, Edge>>(nodes, edges, inputNodes, outputNodes);
        default:
            return std::make_shared<NeuralNetwork<Node, Edge>>(nodes, edges, inputNodes, outputNodes);
        }
    }
};
