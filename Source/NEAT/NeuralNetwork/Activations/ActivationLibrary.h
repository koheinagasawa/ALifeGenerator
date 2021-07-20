/*
* ActivationLibrary.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/NeuralNetwork/Activations/Activation.h>
#include <Common/BaseType.h>

#include <memory>
#include <unordered_map>

DECLARE_ID(ActivationId, uint8_t);

// Library of activation functions.
class ActivationLibrary
{
public:
    using ActivationPtr = std::shared_ptr<Activation>;
    using ActivationMap = std::unordered_map<ActivationId, ActivationPtr>;

    // Constructor.
    ActivationLibrary();

    // Register a new activation function.
    auto registerActivation(ActivationPtr activation)->ActivationId;

    // Unregister an existing activation function.
    void unregisterActivation(ActivationId id);

    // Get an activation function from its id.
    auto getActivation(ActivationId id)->ActivationPtr;

    // Return true if the activation is registered.
    bool hasActivation(ActivationPtr activation) const;

    // Return true if the id is for a registered activation function.
    bool isActivationIdValid(ActivationId id) const;

    // Get the maximum activation id which has ever created by this library.
    inline ActivationId getMaxActivationId() const { return m_nextActivationId.m_val - 1; }

protected:
    ActivationMap m_registry;
    ActivationId m_nextActivationId;
};
