/*
* ActivationLibrary.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/NeuralNetwork/Activations/ActivationLibrary.h>

ActivationLibrary::ActivationLibrary()
    : m_nextActivationId(0)
{
}

auto ActivationLibrary::registerActivation(ActivationPtr activation)->ActivationId
{
    if (!activation)
    {
        return ActivationId::invalid();
    }

    m_registry[m_nextActivationId] = activation;

    ActivationId id = m_nextActivationId;
    m_nextActivationId = m_nextActivationId.m_val + 1;
    return id;
}

void ActivationLibrary::unregisterActivation(ActivationId id)
{
    m_registry.erase(id);
}

auto ActivationLibrary::getActivation(ActivationId id) const->ActivationPtr
{
    if (isActivationIdValid(id))
    {
        return m_registry.at(id);
    }
    else
    {
        return nullptr;
    }
}

bool ActivationLibrary::hasActivation(ActivationPtr activation) const
{
    for (auto& elem : m_registry)
    {
        if (elem.second == activation)
        {
            return true;
        }
    }

    return false;
}

bool ActivationLibrary::isActivationIdValid(ActivationId id) const
{
    return m_registry.find(id) != m_registry.end();
}

auto ActivationLibrary::getActivationIds() const->std::vector<ActivationId>
{
    std::vector<ActivationId> idsOut;
    idsOut.reserve(getNumActivations());
    for (auto& elem : m_registry)
    {
        idsOut.push_back(elem.first);
    }
    return idsOut;
}
