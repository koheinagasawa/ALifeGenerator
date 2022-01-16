/*
* PointBasedSystemSolver.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

// An abstract class of solver for point based system.
class PointBasedSystemSolver
{
public:
    enum class Type
    {
        POSITION_BASED_DYNAMICS,
        MASS_SPRING,
    };

    virtual void solve(float deltaTime) = 0;
    virtual Type getType() const = 0;
};
