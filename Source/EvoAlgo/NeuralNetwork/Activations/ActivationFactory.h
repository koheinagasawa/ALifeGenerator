/*
* ActivationFactory.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <EvoAlgo/NeuralNetwork/Activations/Activation.h>
#include <memory>

// Factory class of predefined activation functions.
class ActivationFacotry
{
public:
    using ActivationPtr = std::shared_ptr<Activation>;

    enum Type
    {
        AF_SIGMOID,
        AF_BIPOLAR_SIGMOID,
        AF_RELU,
        AF_GAUSSIAN,
        AF_ABSOLUTE,
        AF_SINE,
        AF_COSINE,
        AF_TANGENT,
        AF_HYPERBOLIC_TANGENT,
        AF_RAMP,
        AF_STEP,
        AF_SPIKE,
        AF_INVERSE,
        AF_IDENTITY,
        AF_CLAMPED,
        AF_LOGARITHMIC,
        AF_EXPONENTIAL,
        AF_HAT,
        AF_SQUARE,
        AF_CUBE
    };

    // Create an activation of the type.
    static auto create(Type type)->ActivationPtr;
};
