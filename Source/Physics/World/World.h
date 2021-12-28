/*
* World.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Physics/Physics.h>

class System;

// A world of physics simulation.
class World
{
public:
    // Type definitions
    using Systems = std::vector<System*>;

    // Step the world by deltaTime.
    void step(float deltaTime);

    // Add simulation system.
    void addSystem(System& system);

protected:
    // The simulation systems.
    Systems m_systems;
};
