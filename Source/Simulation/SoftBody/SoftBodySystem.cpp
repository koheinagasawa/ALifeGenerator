/*
* SoftBodySystem.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Simulation/Simulation.h>
#include <Simulation/SoftBody/SoftBodySystem.h>
#include <Simulation/SoftBody/Solver/SoftBodySolver.h>

void SoftBodySystem::step(float deltaTime)
{
    assert(m_solver);
    assert(deltaTime > 0.f);

    m_solver->solve(deltaTime);
}
