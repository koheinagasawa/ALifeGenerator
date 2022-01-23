/*
* System.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/Math/Vector4.h>

// An abstract class of simulation system.
class System
{
public:
    // Constructor
    System() = default;

    // Step this system by deltaTime.
    virtual void step(float deltaTime) = 0;
};
