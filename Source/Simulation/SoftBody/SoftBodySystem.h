/*
* SoftBodySystem.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Simulation/System/System.h>
#include <memory>

class SoftBodySolver;

class SoftBodySystem : public System
{
public:
    // Type definitions
    using SolverPtr = std::shared_ptr<SoftBodySolver>;

    virtual void step(float deltaTime) override;

protected:
    SolverPtr m_solver;
};
