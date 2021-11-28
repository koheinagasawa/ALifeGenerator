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
    using SystemPtr = std::shared_ptr<System>;
    using Systems = std::vector<SystemPtr>;

    // Step the world by deltaTime.
    void step(float deltaTime);

    // Add simulation system.
    void addSystem(SystemPtr system);

protected:
    // The simulation systems.
    Systems m_systems;
};
