/*
* MassSpringSolver.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Physics/Solvers/PointBasedSystemSolver.h>
#include <Physics/Collision/Collider.h>
#include <Common/Math/Vector4.h>

class PointBasedSystem;

// Mass spring solver.
class MassSpringSolver : public PointBasedSystemSolver
{
public:
    // Constructor.
    MassSpringSolver(PointBasedSystem& system, float dampingFactor);

    // Step and solve.
    virtual void solve(float deltaTime) override;

    virtual Type getType() const override { return Type::MASS_SPRING; }

    inline auto getDampingFactor() const->const SimdFloat& { return m_dampingFactor; }

protected:
    // Spring constraint.
    struct Constraint
    {
        int m_vertexA, m_vertexB;   // Indices of two vertices connected by this spring.
        SimdFloat m_length;         // The natural length of this spring.
        SimdFloat m_springFactor;   // The spring factor.
    };

    // Type definitions.
    using Positions = std::vector<Vector4>;
    using Velocities = std::vector<Vector4>;
    using Forces = std::vector<Vector4>;
    using Constraints = std::vector<Constraint>;
    using Colliders = std::vector<Collider>;

    Positions& m_positions;         // External buffer of vertex positions.
    Velocities& m_velocities;       // External buffer of vertex velocities.
    Forces m_forces;                // Forces for each vertex.
    Constraints m_constraints;      // The springs.
    const Colliders& m_colliders;   // The colliders.
    Vector4 m_gravity;
    SimdFloat m_dampingFactor;
};
