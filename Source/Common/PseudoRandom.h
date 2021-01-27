/*
* PseudoRandom.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <random>

class RandomGenerator
{
public:

    // Get a random float between 0 and 1.
    virtual float randomReal01() = 0;

    // Get a random float between min and max.
    virtual float randomReal(float min, float max) = 0;

    // Get a random integer between min and max.
    virtual int randomInteger(int min, int max) = 0;

    // Get a random boolean.
    virtual bool randomBoolean() = 0;
};

// Helper class to generate pseudo random number of uniform distribution by mersenne twister.
class PseudoRandom : public RandomGenerator
{
public:
    PseudoRandom(int seed);

    // Get the global random generator.
    static PseudoRandom& getInstance();

    // Get a random float between 0 and 1.
    virtual float randomReal01() override;

    // Get a random float between min and max.
    virtual float randomReal(float min, float max) override;

    // Get a random integer between min and max.
    virtual int randomInteger(int min, int max) override;

    // Get a random boolean.
    virtual bool randomBoolean() override;

protected:
    PseudoRandom(const PseudoRandom&) = delete;
    void operator=(const PseudoRandom&) = delete;

    std::mt19937 m_engine;

    static PseudoRandom s_instance;
};
