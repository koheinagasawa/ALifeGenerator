/*
* ActivationLibrary.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/Activations/Activation.h>
#include <NEAT/NeuralNetwork/Activations/ActivationFactory.h>
#include <Common/BaseType.h>

#include <memory>
#include <unordered_map>

DECLARE_ID(ActivationId, uint8_t);

// Library of activation functions.
class ActivationLibrary
{
public:
    using ActivationPtr = std::shared_ptr<const Activation>;
    using ActivationMap = std::unordered_map<ActivationId, ActivationPtr>;

    // Constructor.
    ActivationLibrary();

    // Register a new activation function.
    auto registerActivation(ActivationPtr activation)->ActivationId;

    // Batch register activations which can be created by ActivationFactory.
    auto registerActivations(const std::vector<ActivationFacotry::Type>& types)->std::vector<ActivationId>;

    // Unregister an existing activation function.
    void unregisterActivation(ActivationId id);

    // Return the number of registered activation functions.
    inline int getNumActivations() const { return (int)m_registry.size(); }

    // Get an activation function from its id.
    auto getActivation(ActivationId id) const->ActivationPtr;

    // Return true if the activation is registered.
    bool hasActivation(ActivationPtr activation) const;

    // Return true if the id is for a registered activation function.
    bool isActivationIdValid(ActivationId id) const;

    // Return a list of activation IDs
    auto getActivationIds() const->std::vector<ActivationId>;

    // Get the maximum activation id which has ever created by this library.
    inline ActivationId getMaxActivationId() const { return m_nextActivationId.m_val - 1; }

protected:
    ActivationMap m_registry;
    ActivationId m_nextActivationId;
};
