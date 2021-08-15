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

    ActivationId id = m_nextActivationId;
    activation->m_id = id;
    m_registry[id] = activation;

    m_nextActivationId = id.m_val + 1;
    return id;
}

auto ActivationLibrary::registerActivations(const std::vector<ActivationFacotry::Type>& types)->std::vector<ActivationId>
{
    std::vector<ActivationId> out;
    out.reserve(types.size());
    for (ActivationFacotry::Type t : types)
    {
        out.push_back(registerActivation(ActivationFacotry::create(t)));
    }

    return out;
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
