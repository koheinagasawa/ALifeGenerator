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
    return randomReal(0.f, 1.f);
}

float PseudoRandom::randomReal(float min, float max)
{
    std::uniform_real_distribution<> dist(min, max);
    return (float)dist(m_engine);
}

int PseudoRandom::randomInteger(int min, int max)
{
    std::uniform_int_distribution<> dist(min, max);
    return dist(m_engine);
}

bool PseudoRandom::randomBoolean()
{
    return randomInteger(0, 9) < 5;
}
