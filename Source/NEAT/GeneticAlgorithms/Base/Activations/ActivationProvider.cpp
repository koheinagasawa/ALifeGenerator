/*
* ActivationProvider.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/Base/Activations/ActivationProvider.h>
#include <NEAT/NeuralNetwork/Activations/ActivationLibrary.h>

//
// DefaultActivationProvider
//

DefaultActivationProvider::DefaultActivationProvider(const Activation* defaultActivation)
    : m_defaultActivation(defaultActivation)
{
}

auto DefaultActivationProvider::getActivation() const->const Activation*
{
    return m_defaultActivation;
}

//
// RandomActivationProvider
//

RandomActivationProvider::RandomActivationProvider(const ActivationLibrary& library, RandomGenerator* random)
    : m_library(library)
    , m_random(random ? random : &PseudoRandom::getInstance())
{
}

auto RandomActivationProvider::getActivation() const->const Activation*
{
    if (m_library.getNumActivations() == 0)
    {
        return nullptr;
    }

    std::vector<ActivationId> activationIds = m_library.getActivationIds();
    ActivationId activationId = activationIds[m_random->randomInteger(0, int(activationIds.size()-1))];
    return m_library.getActivation(activationId).get();
}