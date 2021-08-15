/*
* ActivationFactory.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/Activations/Activation.h>
#include <memory>

// Factory class of predefined activation functions.
class ActivationFacotry
{
public:
    using ActivationPtr = std::shared_ptr<Activation>;

    enum Type
    {
        SIGMOID,
        BIPOLAR_SIGMOID,
        RELU,
        GAUSSIAN,
        ABSOLUTE,
        SINE,
        COSINE,
        TANGENT,
        HYPERBOLIC_TANGENT,
        RAMP,
        STEP,
        SPIKE,
        INVERSE,
        IDENTITY,
        CLAMPED,
        LOGARITHMIC,
        EXPONENTIAL,
        HAT,
        SQUARE,
        CUBE
    };

    // Create an activation of the type.
    static auto create(Type type)->ActivationPtr;
};