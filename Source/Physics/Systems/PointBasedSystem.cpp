/*
* PointBasedSystem.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Physics/Physics.h>
#include <Physics/Systems/PointBasedSystem.h>
#include <Physics/Solvers/PBD/PBDSolver.h>
#include <Physics/Solvers/MassSpring/MassSpringSolver.h>

PointBasedSystem::PointBasedSystem(const Cinfo& cinfo)
{
    m_gravity = cinfo.m_gravity;

    // Create vertices and edges
    {
        const int numVertices = (int)cinfo.m_vertexPositions.size();
        const int numEdges = (int)cinfo.m_vertexConnectivity.size();
        assert(numVertices > 0 && numEdges > 0 && cinfo.m_mass > 0.f);

        // Allocate buffers.
        m_vertices.resize(numVertices);
        m_edges.resize(numEdges);
        m_positions = cinfo.m_vertexPositions;
        m_velocities.resize(numVertices, Vec4_0);

        // Set mass and radius.
        m_vertexMass = cinfo.m_mass / (float)numVertices;
        m_vertexRadius = cinfo.m_radius;

        // Count the number of edges going from each vertex.
        for (int i = 0; i < numEdges; i++)
        {
            const Cinfo::Connection& c = cinfo.m_vertexConnectivity[i];
            assert(c.m_vA >= 0 && c.m_vA < numVertices);
            assert(c.m_vB >= 0 && c.m_vB < numVertices);
            assert(c.m_vA != c.m_vB);

            int vMin = std::min(c.m_vA, c.m_vB);
            m_vertices[vMin].m_numEdges++;
        }

        // Set Vertex data.
        for (int i = 1; i < numVertices; i++)
        {
            Vertex& v = m_vertices[i];
            const Vertex& pv = m_vertices[i - 1];
            v.m_edgeStart = pv.m_edgeStart + pv.m_numEdges;
        }

        // Clear m_numEdges once (needed for the following operation).
        for (int i = 0; i < numVertices; i++)
        {
            m_vertices[i].m_numEdges = 0;
        }

        // Set Edge data and count Vertex.m_numEdges again.
        for (int i = 0; i < numEdges; i++)
        {
            const Cinfo::Connection& c = cinfo.m_vertexConnectivity[i];
            int vA = std::min(c.m_vA, c.m_vB);
            int vB = std::max(c.m_vA, c.m_vB);

            Vertex& v = m_vertices[vA];
            Edge& e = m_edges[v.m_edgeStart + v.m_numEdges];
            e.m_otherVertex = vB;
            e.m_length = (m_positions[vA] - m_positions[vB]).length<3>();
            e.m_stiffness = SimdFloat(c.m_stiffness);
            v.m_numEdges++;
        }
    }

    createSolver(cinfo);
}

void PointBasedSystem::createSolver(const Cinfo& cinfo)
{
    switch (cinfo.m_solverType)
    {
    case PointBasedSystemSolver::Type::POSITION_BASED_DYNAMICS:
    {
        m_solver = std::make_shared<PBD::Solver>(*this, cinfo.m_solverIterations, cinfo.m_dampingFactor);
        break;
    }
    case PointBasedSystemSolver::Type::MASS_SPRING:
    {
        m_solver = std::make_shared<MassSpringSolver>(*this, cinfo.m_dampingFactor);
        break;
    }
    default:
        assert(0);
        break;
    }
}

void PointBasedSystem::step(float deltaTime)
{
    assert(m_solver);

    // Solve
    m_solver->solve(deltaTime);
}

void PointBasedSystem::addCollider(const ShapePtr shape)
{
    m_colliders.push_back(shape);
}
