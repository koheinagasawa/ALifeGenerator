/*
* World.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Physics/Physics.h>
#include <Physics/World/World.h>
#include <Physics/Systems/System.h>

void World::step(float deltaTime)
{
    // Step all the systems.
    for (SystemPtr system : m_systems)
    {
        system->step(deltaTime);
    }
}

void World::addSystem(SystemPtr system)
{
    m_systems.push_back(system);
}
