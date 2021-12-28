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

    // Accessors to gravity.
    inline auto getGravity() const->const Vector4& { return m_gravity; }
    inline void setGravity(const Vector4& g) { m_gravity = g; }

protected:
    Vector4 m_gravity;
};
