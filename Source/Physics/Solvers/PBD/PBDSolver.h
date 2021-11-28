/*
* PBDSolver.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Physics/Solvers/PointBasedSystemSolver.h>
#include <Physics/Collision/Collider.h>
#include <Common/Math/Vector4.h>

#include <vector>

class PointBasedSystem;

// Code for Position Based Dynamics.
namespace PBD
{
    // An abstract constraint type.
    struct Constraint
    {
    public:
        Constraint(SimdFloat stiffness);

        virtual void project() = 0;

        SimdFloat m_stiffness;
    };

    // Constraint for collision between two dynamic vertices.
    struct DynamicVertexCollisionConstraint : public Constraint
    {
    public:
        DynamicVertexCollisionConstraint(
            Vector4* positionA, Vector4* positionB,
            const SimdFloat& radiusA, const SimdFloat& radiusB,
            const SimdFloat& stiffness, int solverIterations);

        virtual void project();

        Vector4* m_positionA;
        Vector4* m_positionB;
        SimdFloat m_radiusSq;
    };

    // Constraint for collision between a vertex and a static collider.
    struct StaticCollisionConstraint : public Constraint
    {
    public:
        StaticCollisionConstraint(
            Vector4* position, const Vector4& target, const Vector4& normal,
            const SimdFloat& stiffness, int solverIterations);

        virtual void project();

        Vector4* m_position;
        Vector4 m_targetPosition;
        Vector4 m_normal;
    };

    // Position Based Dynamic solver
    // [TODO] The solver assumes that mass of all vertices are the same.
    class Solver : public PointBasedSystemSolver
    {
    public:
        // Type definitions
        using Positions = std::vector<Vector4>;
        using Velocities = std::vector<Vector4>;
        using Colliders = std::vector<Collider>;
        using Constraints = std::vector<Constraint*>;
        using DynamicVertexCollisionConstraints = std::vector<DynamicVertexCollisionConstraint>;
        using StaticCollisionConstraints = std::vector<StaticCollisionConstraint>;

        // Type for velocity damping
        enum class VelocityDampingType
        {
            NONE,           // No velocity damping.
            SIMPLE,         // Simple method to damp a certain velocity every frame.
            SHAPE_MATCH     // Apply damping based on shape match in order to maintain the original shape.
        };

        // Constructor
        Solver(PointBasedSystem& system, int solverIterations, float dampingFactor);

        // Destructor
        virtual ~Solver();

        // Step and solve the vertices.
        virtual void solve(float deltaTime) override;

        inline int getNumVertices() const { return (int)m_positions.size(); }
        inline int getNumColliders() const { return (int)m_colliders.size(); }

    protected:
        void dampVelocities();

        void generateCollisionConstraints();

        void projectConstraints();

        Positions& m_positions;     // External buffer of vertex positions.
        Positions m_newPositions;   // Temporary buffer to calculate new vertex positions.

        Velocities& m_velocities;   // External buffer of vertex velocities.

        const Colliders& m_colliders;   // The colliders.

        // Constraints
        Constraints m_constraints;
        DynamicVertexCollisionConstraints m_dynamicVertexCollisionConstraints;
        StaticCollisionConstraints m_staticCollisionConstraints;

        Vector4 m_gravity;

        int m_solverIterations;

        VelocityDampingType m_dampingType = VelocityDampingType::SHAPE_MATCH;
        SimdFloat m_dampingFactor;

        SimdFloat m_vertexRadius;
    };
}
