/*
* NeuralNetworkEvaluator.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/NeuralNetwork.h>

// Helper class to evaluate neural network.
class NeuralNetworkEvaluator
{
public:
    // Method of evaluation
    enum class EvaluationType
    {
        ITERATION, // Perform evaluation iteratively for a certain number of times.
        CONVERGE,  // Perform evaluation until the output values converge.
    };

    template <typename Node, typename Edge>
    void evaluate(NeuralNetwork<Node, Edge>* network) const;

public:
    EvaluationType m_type;
    uint16_t m_evalIterations;
    float m_convergenceThreshold;
};

template <typename Node, typename Edge>
void NeuralNetworkEvaluator::evaluate(NeuralNetwork<Node, Edge>* network) const
{
    if (network->allowsCircularNetwork())
    {
        switch (m_type)
        {
        case NeuralNetworkEvaluator::EvaluationType::ITERATION:
        {
            // TODO
            break;
        }
        case NeuralNetworkEvaluator::EvaluationType::CONVERGE:
            // TODO
            break;
        default:
            assert(0);
            break;
        }
    }
    else
    {
        network->evaluate();
    }
}
