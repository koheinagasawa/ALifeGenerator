/*
* System.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

class SoftBodySolver
{
public:
    virtual void solve(float deltaTime) = 0;
};
