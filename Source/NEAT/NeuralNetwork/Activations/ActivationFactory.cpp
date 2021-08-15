/*
* ActivationFactory.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/NeuralNetwork/Activations/ActivationFactory.h>

#include <algorithm>

auto ActivationFacotry::create(Type type)->ActivationPtr
{
    ActivationPtr out;

    switch (type)
    {
    case SIGMOID:
        out = std::make_shared<Activation>([](float val) { return 1.f / (1.f + expf(-4.9f*val)); });
        out->m_name = "sigmoid";
        break;
    case BIPOLAR_SIGMOID:
        out = std::make_shared<Activation>([](float val) { return (1.f - expf(-val)) / (1.f + expf(-val)); });
        out->m_name = "bipolar sigmoid";
        break;
    case RELU:
        out = std::make_shared<Activation>([](float val) { return std::max(0.f, val); });
        out->m_name = "relu";
        break;
    case GAUSSIAN:
        out = std::make_shared<Activation>([](float val) { return expf(-val * val); });
        out->m_name = "gaussian";
        break;
    case LINEAR:
        out = std::make_shared<Activation>([](float val) { return val; });
        out->m_name = "linear";
        break;
    case ABSOLUTE:
        out = std::make_shared<Activation>([](float val) { return fabsf(val); });
        out->m_name = "abs";
        break;
    case SINE:
        out = std::make_shared<Activation>([](float val) { return sinf(val); });
        out->m_name = "sin";
        break;
    case COSINE:
        out = std::make_shared<Activation>([](float val) { return cosf(val); });
        out->m_name = "cos";
        break;
    case TANGENT:
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
    case HYPERBOLIC_TANGENT:
        out = std::make_shared<Activation>([](float val) { return tanhf(val); });
        out->m_name = "tanh";
        break;
    case RAMP:
        out = std::make_shared<Activation>([](float val) { return 1.0f - 2.0f * (val - floorf(val)); });
        out->m_name = "ramp";
        break;
    case STEP:
        out = std::make_shared<Activation>([](float val)
            {
                return (int)floorf(val)%2 ? -1.0f : 1.0f;
            });
        out->m_name = "step";
        break;
    case SPIKE:
        out = std::make_shared<Activation>([](float val)
            {
                return (int)floorf(val) % 2 ? -1.0f + 2.0f * (val - floorf(val)) : (1.f - 2.0f * (val - floorf(val)));
            });
        out->m_name = "spike";
        break;
    case INVERSE:
        out = std::make_shared<Activation>([](float val) { return 1.0f/val; });
        out->m_name = "inverse";
        break;
    case IDENTITY:
        out = std::make_shared<Activation>([](float val) { return val; });
        out->m_name = "identity";
        break;
    case CLAMPED:
        out = std::make_shared<Activation>([](float val)
            {
                return val < 0.f ? 0.f : (val > 1.f ? 1.f : val);
            });
        out->m_name = "clamped";
        break;
    case LOGARITHMIC:
        out = std::make_shared<Activation>([](float val) { return logf(val); });
        out->m_name = "log";
        break;
    case EXPONENTIAL:
        out = std::make_shared<Activation>([](float val) { return expf(val); });
        out->m_name = "exp";
        break;
    case HAT:
        out = std::make_shared<Activation>([](float val) 
            {
                float valAbs = fabsf(val);
                return valAbs < 1.f ? 1 - valAbs : 0.f;
            });
        out->m_name = "hat";
        break;
    case SQUARE:
        out = std::make_shared<Activation>([](float val) { return val * val; });
        out->m_name = "square";
        break;
    case CUBE:
        out = std::make_shared<Activation>([](float val) { return val * val * val; });
        out->m_name = "cube";
        break;
    default:
        WARN("Invalid activation type.");
        out = nullptr;
        break;
    }

    return out;
}
