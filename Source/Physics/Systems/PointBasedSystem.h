/*
* PointBasedSystem.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Physics/Physics.h>
#include <Physics/Systems/System.h>
#include <Physics/Solvers/PointBasedSystemSolver.h>
#include <Physics/Collision/Collider.h>

#include <functional>
#include <map>

class Shape;

// A system of points connected by edges each other.
class PointBasedSystem : public System
{
public:
    // Vertex data.
    struct Vertex
    {
        int m_edgeStart = 0;    // Start index of edges of this vertex.
        int m_numEdges = 0;     // The number of edges going from this vertex.
    };

    // Edge data.
    struct Edge
    {
        int m_otherVertex;      // Index of the other vertex.
        SimdFloat m_length;     // Default length of this edge.
        SimdFloat m_stiffness;  // Stiffness of this edge.
    };

    // Type Definitions
    using Vertices = std::vector<Vertex>;
    using Positions = std::vector<Vector4>;
    using Edges = std::vector<Edge>;
    using Velocities = std::vector<Vector4>;
    using Colliders = std::vector<Collider>;
    using SolverPtr = std::shared_ptr<PointBasedSystemSolver>;
    using ShapePtr = std::shared_ptr<Shape>;
    using SolverType = PointBasedSystemSolver::Type;
    using OnParticleAddedFunc = std::function<void(const Positions&)>;
    using OnParticleAddedFuncs = std::map<int, OnParticleAddedFunc>; // If we use unordered_map here, we hit a crash by an illegal instruction. It's probably due to a compiler bug.

    // Construction info
    struct Cinfo
    {
        // Connectivity data of points. Edges will be constructed by this.
        struct Connection
        {
            int m_vA, m_vB;             // Indices of vertices.
            float m_stiffness = 1.0f;   // Stiffness of this connection.

            // Length of this connection. If zero, the length will be initial distance of the two vertices.
            float m_length = 0.f;
        };

        using Connections = std::vector<Connection>;

        SolverType m_solverType = SolverType::POSITION_BASED_DYNAMICS;
        int m_solverIterations = 1;

        Positions m_vertexPositions;
        Connections m_vertexConnectivity;
        float m_mass = 1.0f;
        float m_radius = 1.0f;
        float m_dampingFactor = 1.0f;

        Vector4 m_gravity = Vector4{0.f, -9.8f, 0.f};
    };

    // Constructor
    PointBasedSystem() = default;

    // Initialize by Cinfo.
    void init(const Cinfo& cinfo);

    // Add new vertices and edges and remove some edges.
    // edgesToRemove has to be sorted by edgeId in increasing order.
    // [TODO] Should we support to remove vertices too?
    void addRemoveVerticesAndEdges(const Positions& newVertices, const Velocities& newVelocities, const Cinfo::Connections& newEdges, const std::vector<int>& edgesToRemove = std::vector<int>());

    // Step this system by deltaTime.
    virtual void step(float deltaTime) override;

    // Add/remove colliders.
    void addCollider(const ShapePtr shape);
    void removeCollider(const ShapePtr shape);

    // Return mass of each vertex.
    float getVertexMass() const { return m_vertexMass; }

    // Return radius of each vertex.
    float getVertexRadius() const { return m_vertexRadius; }

    // Accessors to simulation data.
    inline const Vertices& getVertices() const { return m_vertices; }
    inline const Edges& getEdges() const { return m_edges; }
    inline const Colliders& getColliders() const { return m_colliders; }
    inline const Positions& getVertexPositions() const { return m_positions; }
    inline const Velocities& getVertexVelocities() const { return m_velocities; }
    inline Positions& accessVertexPositions() { return m_positions; }
    inline Velocities& accessVertexVelocities() { return m_velocities; }

    // Subscribe to on particle added callback. Return handle of the callback.
    int subscribeToOnParticleAdded(const OnParticleAddedFunc& f);

    // Unsubscribe from on particle added callback.
    void unsubscribeFromOnParticleAdded(int handle);

protected:
    void createSolver(const Cinfo& cinfo);
    void updateSolver();

    void onParticlesAdded(const Positions& posOfNewVertices) const;

    Vertices m_vertices;        // The vertices
    Edges m_edges;              // The vertex edges.
    Positions m_positions;      // Positions of the vertices.
    Velocities m_velocities;    // Velocities of the vertices.

    float m_vertexMass;     // Mass of each vertex.
    float m_vertexRadius;   // Radius of each vertex.

    Colliders m_colliders;  // The colliders.

    SolverPtr m_solver;     // The solver.

    OnParticleAddedFuncs m_onParticleAddedFuncs; // List of callback function called after new particles were added to this system.
};
