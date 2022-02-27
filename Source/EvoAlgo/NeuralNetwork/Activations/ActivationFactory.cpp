/*
* ActivationFactory.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <EvoAlgo/EvoAlgo.h>
#include <EvoAlgo/NeuralNetwork/Activations/ActivationFactory.h>

#include <algorithm>

#define FLOAT_HIGH 1E+10f

inline float clamp(float v)
{
    return std::max(-FLOAT_HIGH, std::min(FLOAT_HIGH, v));
}

auto ActivationFacotry::create(Type type)->ActivationPtr
{
    ActivationPtr out;

    switch (type)
    {
    case AF_SIGMOID:
        out = std::make_shared<Activation>([](float val) { return 1.f / (1.f + expf(-4.9f*val)); });
        out->m_name = "sigmoid";
        break;
    case AF_BIPOLAR_SIGMOID:
        out = std::make_shared<Activation>([](float val) { return clamp(1.f - expf(-val)) / clamp(1.f + expf(-val)); });
        out->m_name = "bipolar sigmoid";
        break;
    case AF_RELU:
        out = std::make_shared<Activation>([](float val) { return std::max(0.f, val); });
        out->m_name = "relu";
        break;
    case AF_GAUSSIAN:
        out = std::make_shared<Activation>([](float val) { return clamp(-val * val); });
        out->m_name = "gaussian";
        break;
    case AF_ABSOLUTE:
        out = std::make_shared<Activation>([](float val) { return fabsf(val); });
        out->m_name = "abs";
        break;
    case AF_SINE:
        out = std::make_shared<Activation>([](float val) { return sinf(val); });
        out->m_name = "sin";
        break;
    case AF_COSINE:
        out = std::make_shared<Activation>([](float val) { return cosf(val); });
        out->m_name = "cos";
        break;
    case AF_TANGENT:
        out = std::make_shared<Activation>([](float val)
            {
                constexpr float max = 10000.0f;
                val = tanf(val);
                if (val < max && val >-max) return val;
                else if (val >= max) return max;
                else return -max;
            });
        out->m_name = "tan";
        break;
    case AF_HYPERBOLIC_TANGENT:
        out = std::make_shared<Activation>([](float val) { return tanhf(val); });
        out->m_name = "tanh";
        break;
    case AF_RAMP:
        out = std::make_shared<Activation>([](float val) { return 1.0f - 2.0f * (val - floorf(val)); });
        out->m_name = "ramp";
        break;
    case AF_STEP:
        out = std::make_shared<Activation>([](float val)
            {
                return (int)floorf(val)%2 ? -1.0f : 1.0f;
            });
        out->m_name = "step";
        break;
    case AF_SPIKE:
        out = std::make_shared<Activation>([](float val)
            {
                return (int)floorf(val) % 2 ? -1.0f + 2.0f * (val - floorf(val)) : (1.f - 2.0f * (val - floorf(val)));
            });
        out->m_name = "spike";
        break;
    case AF_INVERSE:
        out = std::make_shared<Activation>([](float val) { return clamp(1.0f/val); });
        out->m_name = "inverse";
        break;
    case AF_IDENTITY:
        out = std::make_shared<Activation>([](float val) { return val; });
        out->m_name = "identity";
        break;
    case AF_CLAMPED:
        out = std::make_shared<Activation>([](float val)
            {
                return val < 0.f ? 0.f : (val > 1.f ? 1.f : val);
            });
        out->m_name = "clamped";
        break;
    case AF_LOGARITHMIC:
        out = std::make_shared<Activation>([](float val) { return clamp(logf(val)); });
        out->m_name = "log";
        break;
    case AF_EXPONENTIAL:
        out = std::make_shared<Activation>([](float val) { return clamp(expf(val)); });
        out->m_name = "exp";
        break;
    case AF_HAT:
        out = std::make_shared<Activation>([](float val) 
            {
                float valAbs = fabsf(val);
                return valAbs < 1.f ? 1 - valAbs : 0.f;
            });
        out->m_name = "hat";
        break;
    case AF_SQUARE:
        out = std::make_shared<Activation>([](float val) { return clamp(val * val); });
        out->m_name = "square";
        break;
    case AF_CUBE:
        out = std::make_shared<Activation>([](float val) { return clamp(val * val * val); });
        out->m_name = "cube";
        break;
    default:
        WARN("Invalid activation type.");
        out = nullptr;
        break;
    }

    return out;
}
