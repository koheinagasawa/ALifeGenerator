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
    EvaluationType m_type = EvaluationType::ITERATION;  // The method to evaluate network.
    uint16_t m_evalIterations = 10;                     // The maximum number of iteration to run network.
    float m_convergenceThreshold = 1E-3f;               // Threshold of convergence of output values. Only used for CONVERGE type.
};

template <typename Node, typename Edge>
void NeuralNetworkEvaluator::evaluate(NeuralNetwork<Node, Edge>* network) const
{
    if (network->allowsCircularNetwork())
    {
        // Network containing recursion.
        const bool checkConvergence = m_type == EvaluationType::CONVERGE;

        const NeuralNetwork<Node, Edge>::NodeIds& outputNodes = network->getOutputNodes();
        const int numOutputNodes = (int)outputNodes.size();

        // Prepare a buffer to store output values of the previous evaluation.
        std::vector<float> previousOutputVals;
        if (checkConvergence)
        {
            previousOutputVals.resize(numOutputNodes);
        }

        // Run evaluation multiple times.
        for (uint16_t itr = 0; itr < m_evalIterations; itr++)
        {
            network->evaluate();

            if (checkConvergence)
            {
                if (itr > 0)
                {
                    // Check if the output values have been converged.
                    bool converged = true;
                    for (int i = 0; i < numOutputNodes; i++)
                    {
                        const float nodeVal = network->getNode(outputNodes[i]).getValue();
                        converged &= (std::fabs(previousOutputVals[i] - nodeVal) <= m_convergenceThreshold);
                        previousOutputVals[i] = nodeVal;
                    }

                    if (converged)
                    {
                        break;
                    }
                }
                else
                {
                    // Just copy the output values for the first run.
                    for (int i = 0; i < numOutputNodes; i++)
                    {
                        previousOutputVals[i] = network->getNode(outputNodes[i]).getValue();
                    }
                }
            }
        }
    }
    else
    {
        // Feed forward network. Just evaluate only once.
        network->evaluate();
    }
}
