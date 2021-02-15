/*
* PseudoRandom.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Common/Common.h>
#include <Common/PseudoRandom.h>

PseudoRandom PseudoRandom::s_instance(0);

PseudoRandom& PseudoRandom::getInstance()
{
    return s_instance;
}

PseudoRandom::PseudoRandom(int seed)
    : m_engine(seed)
{
}

float PseudoRandom::randomReal01()
{
    float v = randomReal(0.f, std::nexttoward(1.f, std::numeric_limits<float>::max()));
    // Make sure that v is [0, 1] range. std::uniform_real_distribution should return a value in
    // [min, max) range but it seems it does include max value.
    assert(v <= 1.0f);
    return v;
}

float PseudoRandom::randomReal(float min, float max)
{
    std::uniform_real_distribution<float> dist(min, max);
    return dist(m_engine);
}

int PseudoRandom::randomInteger(int min, int max)
{
    std::uniform_int_distribution<> dist(min, max);
    return dist(m_engine);
}

bool PseudoRandom::randomBoolean()
{
    return randomInteger(0, 1);
}
