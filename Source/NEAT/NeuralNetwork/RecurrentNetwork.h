/*
* RecurrentNetwork.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/NeuralNetwork.h>

#include <set>

template <typename Node, typename Edge>
class RecurrentNetwork : public NeuralNetwork<Node, Edge>
{
public:
    // Type Declarations
    using Base = NeuralNetwork<Node, Edge>;
    using Nodes = Base::Nodes;
    using Edges = Base::Edges;
    using NodeIds = Base::NodeIds;
    using EdgeIds = Base::EdgeIds;
    using NodeData = Base::NodeData;
    using NodeDatas = Base::NodeDatas;

    // Constructor from network information
    RecurrentNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& inputNodes, const NodeIds& outputNodes);

    // Create a copy of this network.
    auto clone() const->std::shared_ptr<NeuralNetwork<Node, Edge>> override;

    // Return type of this network.
    virtual NeuralNetworkType getType() const override { return NeuralNetworkType::RECURRENT; }
};

template <typename Node, typename Edge>
RecurrentNetwork<Node, Edge>::RecurrentNetwork(const Nodes& nodes, const Edges& edges, const NodeIds& inputNodes, const NodeIds& outputNodes)
    : Base(nodes, edges, inputNodes, outputNodes)
{
}

template <typename Node, typename Edge>
auto RecurrentNetwork<Node, Edge>::clone() const->std::shared_ptr<NeuralNetwork<Node, Edge>>
{
    return std::make_shared<RecurrentNetwork>(*this);
}
