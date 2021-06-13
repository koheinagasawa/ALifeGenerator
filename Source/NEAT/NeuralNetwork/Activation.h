/*
* Activation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <functional>
#include <string>

// Wrapper struct for activation function.
struct Activation
{
    using Func = std::function<float(float)>;

    Activation(const Func& func) : m_func(func) {}

    float activate(float value) const { return m_func(value); }

    std::string m_name;
    const Func m_func;
};
