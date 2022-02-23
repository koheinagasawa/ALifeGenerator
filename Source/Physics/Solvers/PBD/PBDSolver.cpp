/*
* PBDSolver.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Physics/Physics.h>
#include <Physics/Solvers/PBD/PBDSolver.h>
#include <Physics/Systems/PointBasedSystem.h>
#include <Physics/Collision/Collider.h>
#include <Physics/Solvers/PBD/Constraints/PBDConstraints.h>
#include <Common/Math/Matrix33.h>

namespace PBD
{
    //
    // Constraint
    //

    Constraint::Constraint(SimdFloat stiffness)
        : m_stiffness(stiffness)
    {
    }

    //
    // DynamicVertexCollisionConstraint
    //

    DynamicVertexCollisionConstraint::DynamicVertexCollisionConstraint(
        Vector4* positionA, Vector4* positionB,
        const SimdFloat& radiusA, const SimdFloat& radiusB,
        const SimdFloat& stiffness, int solverIterations)
        : Constraint(stiffness)
        , m_positionA(positionA), m_positionB(positionB), m_radiusSq(radiusA + radiusB)
    {
        m_radiusSq *= m_radiusSq;
    }

    void DynamicVertexCollisionConstraint::project()
    {
        Vector4 dir = *m_positionA - *m_positionB;
        SimdFloat distSq = dir.lengthSq<3>();
        if (distSq < m_radiusSq)
        {
            if (distSq.getFloat() >= std::numeric_limits<float>::epsilon())
            {
                dir.normalize<3>();
            }
            else
            {
                // Two particles are completely overlapping.
                // Move them towards arbitrary direction.
                // [TODO] Come up with a better solution.
                dir = Vec4_1000;
            }

            SimdFloat halfDiff = SimdFloat((sqrtf(m_radiusSq.getFloat()) - sqrtf(distSq.getFloat())) * 0.5f);
            dir *= halfDiff;
            *m_positionA += dir;
            *m_positionB -= dir;

            assert(!isnan((*m_positionA)(0)) && !isnan((*m_positionA)(1)) && !isnan((*m_positionA)(2)));
            assert(!isnan((*m_positionB)(0)) && !isnan((*m_positionB)(1)) && !isnan((*m_positionB)(2)));
        }
    }

    //
    // StaticCollisionConstraint
    //

    StaticCollisionConstraint::StaticCollisionConstraint(
        Vector4* position, const Vector4& target, const Vector4& normal,
        const SimdFloat& stiffness, int solverIterations)
        : Constraint(stiffness)
        , m_position(position), m_targetPosition(target), m_normal(normal)
    {
    }

    void StaticCollisionConstraint::project()
    {
        if ((*m_position - m_targetPosition).dot<3>(m_normal) < SimdFloat_0)
        {
            *m_position = m_targetPosition;

            assert(!isnan((*m_position)(0)) && !isnan((*m_position)(1)) && !isnan((*m_position)(2)));
        }
    }

    inline float getAdjustedStiffness(const SimdFloat& stiffness, int solverIterations)
    {
        return 1.0f - powf(1.0f - stiffness.getFloat(), 1.0f / (float)solverIterations);
    }

    //
    // Solver
    //

    Solver::Solver(PointBasedSystem& system, const Vector4& gravity, int solverIterations, float dampingFactor)
        : m_positions(system.accessVertexPositions())
        , m_velocities(system.accessVertexVelocities())
        , m_colliders(system.getColliders())
        , m_gravity(gravity)
        , m_solverIterations(solverIterations)
        , m_dampingFactor(dampingFactor)
        , m_vertexRadius(system.getVertexRadius())
    {
        m_newPositions.resize(m_positions.size());

        // Create stretch constraints at all edges between vertices in the point based system.
        {
            SimdFloat mass(system.getVertexMass());

            const PointBasedSystem::Vertices& vertices = system.getVertices();
            const PointBasedSystem::Edges& edges = system.getEdges();

            m_constraints.reserve(edges.size());

            const int numVerts = getNumVertices();

            for (int vtxIdx = 0; vtxIdx < numVerts; vtxIdx++)
            {
                const PointBasedSystem::Vertex& vertex = vertices[vtxIdx];
                if(vertex.m_numEdges > 0)
                {
                    const int edgeEnd = vertex.m_edgeStart + vertex.m_numEdges;

                    for (int edgeIdx = vertex.m_edgeStart; edgeIdx < edgeEnd; edgeIdx++)
                    {
                        const PointBasedSystem::Edge& edge = edges[edgeIdx];

                        SimdFloat stiffness = SimdFloat(getAdjustedStiffness(edge.m_stiffness, m_solverIterations));

                        // Create stretch constraint
                        m_constraints.push_back(new StretchConstraint(
                            SimdFloat(edge.m_length), &m_newPositions[vtxIdx], &m_newPositions[edge.m_otherVertex],
                            mass, mass, stiffness));
                    }
                }
            }
        }
    }

    Solver::~Solver()
    {
        // Clear constraint
        for (int i = 0; i < (int)m_constraints.size(); i++)
        {
            delete m_constraints[i];
        }
        m_constraints.clear();
    }

    void Solver::solve(float deltaTimeIn)
    {
        const SimdFloat dt(deltaTimeIn);

        // Apply gravity
        if (dt > SimdFloat_0 && m_gravity.lengthSq<3>() > SimdFloat_0)
        {
            Vector4 deltaV = m_gravity * dt;
            for (Vector4& vel : m_velocities)
            {
                vel += deltaV;
            }
        }

        // Apply velocity damping
        dampVelocities();

        const int numVertices = getNumVertices();

        assert(numVertices == (int)m_velocities.size());
        assert(numVertices == (int)m_newPositions.size());

        // Move vertices to the new positions.
        for (int i = 0; i < numVertices; i++)
        {
            m_newPositions[i] = m_positions[i] + dt * m_velocities[i];
        }

        // Generate constraints due to collisions.
        generateCollisionConstraints();

        // Project all constraints and repeat.
        for (int i = 0; i < m_solverIterations; i++)
        {
            projectConstraints();
        }

        // Update velocities and positions of vertices.
        const SimdFloat invDt(1.0f / deltaTimeIn);
        for (int i = 0; i < numVertices; i++)
        {
            m_velocities[i] = (m_newPositions[i] - m_positions[i]) * invDt;
            m_positions[i] = m_newPositions[i];
        }
    }

    void Solver::dampVelocities()
    {
        switch (m_dampingType)
        {
            case VelocityDampingType::SHAPE_MATCH:
            {
                const int numVerts = getNumVertices();
                Vector4 avgPos = Vec4_0;
                Vector4 avgVel = Vec4_0;

                for (int i = 0; i < numVerts; i++)
                {
                    avgPos += m_positions[i];
                    avgVel += m_velocities[i];
                }

                SimdFloat invNumVerts(1.f / (float)numVerts);

                avgPos *= invNumVerts;
                avgVel *= invNumVerts;

                std::vector<Vector4> r;
                r.resize(numVerts);
                Vector4 l = Vec4_0;
                Matrix33 inertia = Mat33_0;
                for (int i = 0; i < numVerts; i++)
                {
                    Vector4& ri = r[i];
                    ri = m_positions[i] - avgPos;
                    l += Vector4::cross(ri, m_velocities[i]);

                    Matrix33 m;
                    m.setColumn<0>(Vector4(SimdFloat_0, ri.getComponent<2>(), -ri.getComponent<1>()));
                    m.setColumn<1>(Vector4(-ri.getComponent<2>(), SimdFloat_0, ri.getComponent<0>()));
                    m.setColumn<2>(Vector4(ri.getComponent<1>(), -ri.getComponent<0>(), SimdFloat_0));

                    inertia += ((m * m.transpose()));
                }

                if (inertia.getDeterminant() == SimdFloat_0)
                {
                    break;
                }

                inertia.setInverse(inertia);

                Vector4 angVel = inertia * l;

                for (int i = 0; i < numVerts; i++)
                {
                    m_velocities[i] += m_dampingFactor * (avgVel + Vector4::cross(angVel, r[i]) - m_velocities[i]);
                }
                break;
            }
            case VelocityDampingType::SIMPLE:
            {
                const int numVerts = getNumVertices();
                for (int i = 0; i < numVerts; i++)
                {
                    m_velocities[i] *= (SimdFloat_1 - m_dampingFactor);
                }
                break;
            }
            default:
                break;
        }
    }

    void Solver::generateCollisionConstraints()
    {
        // Detect collision by brute force O(N^2) approach
        // [TODO] Implement a smarter way

        m_dynamicVertexCollisionConstraints.clear();
        m_staticCollisionConstraints.clear();

        const int numVerts = getNumVertices();

        // Find collisions against colliders.
        {
            const int numColliders = getNumColliders();
            for (int posIdx = 0; posIdx < numVerts; posIdx++)
            {
                const Vector4& start = m_positions[posIdx];
                const Vector4& end = m_newPositions[posIdx];

                for (int colIdx = 0; colIdx < numColliders; colIdx++)
                {
                    Shape::RayCastOutput rcOut;
                    const Shape* shape = m_colliders[colIdx].getShape();

                    if (shape->hasInterior())
                    {
                        shape->castRay(start, end, rcOut);
                        if (rcOut.m_hit)
                        {
                            if (rcOut.m_fraction > 0.f)
                            {
                                m_staticCollisionConstraints.push_back(StaticCollisionConstraint(&m_newPositions[posIdx], rcOut.m_hitPoint, rcOut.m_hitNormal, SimdFloat_1, m_solverIterations));
                            }
                            else
                            {
                                Shape::ClosestPointOutput cpOut;
                                shape->getClosestPoint(start, cpOut);
                                m_staticCollisionConstraints.push_back(StaticCollisionConstraint(&m_newPositions[posIdx], cpOut.m_closestPoint, cpOut.m_normal, SimdFloat_1, m_solverIterations));
                            }
                        }
                    }
                    else
                    {
                        Shape::ClosestPointOutput cpOut;
                        shape->getClosestPoint(start, cpOut);
                        Vector4 dir = cpOut.m_closestPoint - start;
                        if (dir.dot<3>(cpOut.m_normal) > SimdFloat_0)
                        {
                            m_staticCollisionConstraints.push_back(StaticCollisionConstraint(&m_newPositions[posIdx], cpOut.m_closestPoint, cpOut.m_normal, SimdFloat_1, m_solverIterations));
                        }
                    }
                }
            }
        }

        // Find collisions between vertices.
        {
            SimdFloat minDistSq = SimdFloat_2 * m_vertexRadius;
            minDistSq *= minDistSq;
            for (int posAIdx = 0; posAIdx < numVerts - 1; posAIdx++)
            {
                Vector4& posA = m_newPositions[posAIdx];
                for (int posBIdx = posAIdx + 1; posBIdx < numVerts; posBIdx++)
                {
                    Vector4& posB = m_newPositions[posBIdx];

                    if ((posA - posB).lengthSq<3>() < minDistSq)
                    {
                        m_dynamicVertexCollisionConstraints.push_back(DynamicVertexCollisionConstraint(&posA, &posB, m_vertexRadius, m_vertexRadius, SimdFloat_1, m_solverIterations));
                    }
                }
            }
        }
    }

    void Solver::projectConstraints()
    {
        // Solve input constraints.
        for (Constraint* c : m_constraints)
        {
            c->project();
        }

        // Solve dynamic collision constraints.
        for (Constraint& c : m_dynamicVertexCollisionConstraints)
        {
            c.project();
        }

        // Solve static collision constraints.
        for (Constraint& c : m_staticCollisionConstraints)
        {
            c.project();
        }
    }
}
