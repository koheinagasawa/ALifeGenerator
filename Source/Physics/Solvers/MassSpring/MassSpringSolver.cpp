/*
* MassSpringSolver.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Physics/Physics.h>
#include <Physics/Solvers/MassSpring/MassSpringSolver.h>
#include <Physics/Systems/PointBasedSystem.h>

MassSpringSolver::MassSpringSolver(PointBasedSystem& system, const Vector4& gravity, float dampingFactor)
    : m_positions(system.accessVertexPositions())
    , m_velocities(system.accessVertexVelocities())
    , m_colliders(system.getColliders())
    , m_gravity(gravity)
    , m_dampingFactor(dampingFactor)
{
    const int numVertices = (int)m_positions.size();
    m_forces.resize(numVertices);

    const PointBasedSystem::Vertices& vertices = system.getVertices();
    const PointBasedSystem::Edges& edges = system.getEdges();

    for (int vtxIdx = 0; vtxIdx < numVertices; vtxIdx++)
    {
        const PointBasedSystem::Vertex& vertex = vertices[vtxIdx];
        if (vertex.m_numEdges > 0)
        {
            const int edgeEnd = vertex.m_edgeStart + vertex.m_numEdges;

            for (int edgeIdx = vertex.m_edgeStart; edgeIdx < edgeEnd; edgeIdx++)
            {
                const PointBasedSystem::Edge& edge = edges[edgeIdx];

                // Create a spring
                m_constraints.push_back(Constraint{vtxIdx, edge.m_otherVertex, edge.m_length, edge.m_stiffness});
            }
        }
    }
}

void MassSpringSolver::solve(float deltaTimeIn)
{
    const int numVertices = (int)m_positions.size();

    for (int i = 0; i < numVertices; i++)
    {
        m_forces[i] = m_gravity;
    }

    const int numConstraints = (int)m_constraints.size();
    for (int i = 0; i < numConstraints; i++)
    {
        const Constraint& c = m_constraints[i];
        const Vector4& posA = m_positions[c.m_vertexA];
        const Vector4& posB = m_positions[c.m_vertexB];
        Vector4 aToB = posB - posA;

        const SimdFloat length = aToB.length<3>();

        aToB.normalize<3>();
        SimdFloat velDiff = aToB.dot<3>(m_velocities[c.m_vertexA] - m_velocities[c.m_vertexB]);

        const SimdFloat factor = (length - c.m_length) * c.m_springFactor - m_dampingFactor * velDiff;

        Vector4& forceA = m_forces[c.m_vertexA];
        Vector4& forceB = m_forces[c.m_vertexB];
        forceA += factor * aToB;
        forceB -= factor * aToB;
    }

    const int numColliders = (int)m_colliders.size();
    const SimdFloat k = SimdFloat(5000.f);
    for (int vi = 0; vi < numVertices; vi++)
    {
        Vector4& pos = m_positions[vi];
        for (int ci = 0; ci < numColliders; ci++)
        {
            const Shape* shape = m_colliders[ci].getShape();

            Shape::ClosestPointOutput cpOut;
            shape->getClosestPoint(pos, cpOut);
            Vector4 dir = cpOut.m_closestPoint - pos;
            if (dir.dot<3>(cpOut.m_normal) > SimdFloat_0)
            {
                Vector4 uDir = dir;
                uDir.normalize<3>();
                Vector4& force = m_forces[vi];
                force += k * dir;
            }
        }
    }

    SimdFloat dt(deltaTimeIn);
    for (int i = 0; i < numVertices; i++)
    {
        m_velocities[i] += m_forces[i] * dt;
        m_positions[i] += m_velocities[i] * dt;
    }
}
