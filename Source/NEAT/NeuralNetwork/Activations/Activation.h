/*
* Activation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/BaseType.h>
#include <functional>
#include <string>

DECLARE_ID(ActivationId, uint8_t);

// Wrapper struct for activation function.
struct Activation
{
    using Func = std::function<float(float)>;

    Activation(const Func& func) : m_func(func) {}

    float activate(float value) const { return m_func(value); }

    const char* m_name;
    const Func m_func;
    ActivationId m_id = 0;
};
