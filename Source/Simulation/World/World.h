/*
* World.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <vector>
#include <memory>

class System;

class World
{
public:
    // Type definitions
    using SystemPtr = std::shared_ptr<System>;
    using Systems = std::vector<SystemPtr>;

    void step(float deltaTime);

protected:
    Systems m_systems;
};
