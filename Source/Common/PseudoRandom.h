/*
* PseudoRandom.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <random>

// Helper class to generate pseudo random random number of uniform distribution.
class PseudoRandom
{
public:
    PseudoRandom(int seed);

    // Get the global random generator.
    static PseudoRandom& getInstance();

    // Get a random float between 0 and 1.
    float randomReal01();

    // Get a random float between min and max.
    float randomReal(float min, float max);

    // Get a random integer between min and max.
    int randomInteger(int min, int max);

    // Get a random boolean.
    bool randomBoolean();

protected:
    PseudoRandom(const PseudoRandom&) = delete;
    void operator=(const PseudoRandom&) = delete;

    std::mt19937 m_engine;

    static PseudoRandom s_instance;
};
