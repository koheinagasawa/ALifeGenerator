/*
* ActivationProvider.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <EvoAlgo/NeuralNetwork/Activations/Activation.h>
#include <Common/PseudoRandom.h>

class ActivationLibrary;

// Base class which provides activation function.
class ActivationProvider
{
public:
    // Provides an activation function.
    virtual auto getActivation() const->const Activation* = 0;
};

// Activation provider which always gives a single default activation function.
class DefaultActivationProvider : public ActivationProvider
{
public:
    // Construct from Activation.
    DefaultActivationProvider(const Activation& defaultActivation);

    // Construct from activation function.
    DefaultActivationProvider(const Activation::Func& func, const char* name = "activation");

    // Provides the default activation function.
    virtual auto getActivation() const->const Activation* override;

protected:
    Activation m_defaultActivation;
};

// Activation provider which gives a random activation from library.
class RandomActivationProvider : public ActivationProvider
{
public:
    RandomActivationProvider(const ActivationLibrary& library, RandomGenerator* random = nullptr);

    // Provides a random activation function from the library.
    virtual auto getActivation() const->const Activation* override;

protected:
    const ActivationLibrary& m_library;
    RandomGenerator* m_random;
};
