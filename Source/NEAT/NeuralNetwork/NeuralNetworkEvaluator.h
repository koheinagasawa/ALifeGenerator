/*
* NeuralNetworkEvaluator.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/NeuralNetwork.h>

class BakedNeuralNetwork;

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

    // Evaluate the given network.
    template <typename Network>
    void evaluate(const std::vector<NodeId>& outputNodes, Network* network) const;

    inline int getCurrentIteration() const { return m_currentItration; }

protected:
    template <typename Network>
    inline bool isCircularNetwork(const Network* network) const;

    template <typename Network>
    inline float getNodeValue(NodeId nodeId, const Network* network) const;

public:
    EvaluationType m_type = EvaluationType::ITERATION;  // The method to evaluate network.
    int m_evalIterations = 10;                          // The maximum number of iteration to run network.
    float m_convergenceThreshold = 1E-3f;               // Threshold of convergence of output values. Only used for CONVERGE type.

protected:
    mutable int m_currentItration;
};

template <typename NeuralNetwork>
inline bool NeuralNetworkEvaluator::isCircularNetwork(const NeuralNetwork* network) const
{
    return network->allowsCircularNetwork();
}

template <>
inline bool NeuralNetworkEvaluator::isCircularNetwork<BakedNeuralNetwork>(const BakedNeuralNetwork* network) const
{
    return network->isCircularNetwork();
}

template <typename NeuralNetwork>
inline float NeuralNetworkEvaluator::getNodeValue(NodeId nodeId, const NeuralNetwork* network) const
{
    return network->getNode(nodeId).getValue();
}

template <>
inline float NeuralNetworkEvaluator::getNodeValue<BakedNeuralNetwork>(NodeId nodeId, const BakedNeuralNetwork* network) const
{
    return network->getNodeValue(nodeId);
}

template <typename Network>
void NeuralNetworkEvaluator::evaluate(const std::vector<NodeId>& outputNodes, Network* network) const
{
    m_currentItration = 0;

    if (isCircularNetwork(network))
    {
        // Network containing recursion.
        const bool checkConvergence = m_type == EvaluationType::CONVERGE;

        const int numOutputNodes = (int)outputNodes.size();

        // Prepare a buffer to store output values of the previous evaluation.
        std::vector<float> previousOutputVals;
        if (checkConvergence)
        {
            previousOutputVals.resize(numOutputNodes);
        }

        // Run evaluation multiple times.
        for (; m_currentItration < m_evalIterations; m_currentItration++)
        {
            network->evaluate();

            if (checkConvergence)
            {
                if (m_currentItration > 0)
                {
                    // Check if the output values have been converged.
                    bool converged = true;
                    for (int i = 0; i < numOutputNodes; i++)
                    {
                        const float nodeVal = getNodeValue(outputNodes[i], network);
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
                        previousOutputVals[i] = getNodeValue(outputNodes[i], network);
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
