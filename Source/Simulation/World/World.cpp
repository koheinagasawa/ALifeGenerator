/*
* World.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Simulation/Simulation.h>
#include <Simulation/World/World.h>
#include <Simulation/System/System.h>

void World::step(float deltaTime)
{
    for (SystemPtr system : m_systems)
    {
        system->step(deltaTime);
    }
}
