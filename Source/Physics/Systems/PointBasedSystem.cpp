/*
* PointBasedSystem.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <Physics/Physics.h>
#include <Physics/Systems/PointBasedSystem.h>
#include <Physics/Solvers/PBD/PBDSolver.h>
#include <Physics/Solvers/MassSpring/MassSpringSolver.h>

void PointBasedSystem::init(const Cinfo& cinfo)
{
    // Create vertices and edges
    const int numVertices = (int)cinfo.m_vertexPositions.size();
    const int numEdges = (int)cinfo.m_vertexConnectivity.size();
    assert(numVertices > 0 && cinfo.m_mass > 0.f);

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
        e.m_length = (c.m_length > 0.f) ? SimdFloat(c.m_length) : (m_positions[vA] - m_positions[vB]).length<3>();
        e.m_stiffness = SimdFloat(c.m_stiffness);
        v.m_numEdges++;
    }

    createSolver(cinfo);

    onParticlesAdded(cinfo.m_vertexPositions);
}

void PointBasedSystem::addRemoveVerticesAndEdges(const Positions& newVertices, const Velocities& newVelocities, const Cinfo::Connections& newEdges, const std::vector<int>& edgesToRemove)
{
    assert(newVertices.size() == newVelocities.size());

    // Preserve previous vertices and edges.
    Vertices prevVerts = m_vertices;
    Edges prevEdges = m_edges;

    const int prevNumVerts = (int)m_positions.size();
    const int newNumVerts = (int)(prevNumVerts + newVertices.size());
    const int numNewEdges = (int)newEdges.size();
    const int prevNumEdges = (int)m_edges.size();
    const int numEdgesToRemove = (int)edgesToRemove.size();

    // Reserve buffers to store newly added vertices and edges.
    m_vertices.reserve(newNumVerts);
    m_positions.reserve(newNumVerts);
    m_velocities.reserve(newNumVerts);
    m_edges.resize(prevNumEdges + numNewEdges - numEdgesToRemove);

    // Add new vertices and velocities
    for (int i = 0; i < (int)newVertices.size(); i++)
    {
        m_positions.push_back(newVertices[i]);
        m_vertices.push_back(Vertex{0, 0});
        m_velocities.push_back(newVelocities[i]);
    }

    // Adjust the number of edges on existing vertices
    if(edgesToRemove.size() > 0)
    {
        assert(prevNumVerts > 0);
        assert(prevNumEdges >= numEdgesToRemove);

        int vertexIndex = 0;
        for (int i = 0; i < numEdgesToRemove; i++)
        {
            assert(i == 0 || edgesToRemove[i - 1] < edgesToRemove[i]); // Make sure the edgesToRemove is sorted by edgeId.

            // Find the vertex where this edge belongs to.
            while (m_vertices[vertexIndex + 1].m_edgeStart <= edgesToRemove[i])
            {
                vertexIndex++;
                assert(vertexIndex < prevNumVerts);
            }
            // Reduce the number of edges in the vertex.
            m_vertices[vertexIndex].m_numEdges--;
        }
    }

    // Count the number of edges per vertex by adding new edges.
    for (const Cinfo::Connection& c : newEdges)
    {
        m_vertices[std::min(c.m_vA, c.m_vB)].m_numEdges++;
    }

    // Calculate start index of edges for each vertex.
    for (int i = 1; i < newNumVerts; i++)
    {
        const Vertex& pv = m_vertices[i - 1];
        m_vertices[i].m_edgeStart = pv.m_edgeStart + pv.m_numEdges;
    }

    // Clear m_numEdges once (needed for the following operation).
    for (int i = 0; i < newNumVerts; i++)
    {
        m_vertices[i].m_numEdges = 0;
    }

    // Move existing edges to new positions.
    int removeEdgeListIdx = 0;
    for (int vA = 0; vA < prevNumVerts; vA++)
    {
        const Vertex& prevVertex = prevVerts[vA];
        for (int i = 0; i < prevVertex.m_numEdges; i++)
        {
            if (removeEdgeListIdx < numEdgesToRemove && (prevVertex.m_edgeStart + i == edgesToRemove[removeEdgeListIdx]))
            {
                // Skip this edge
                removeEdgeListIdx++;
                continue;
            }

            // Move this edge to a new position in the buffer.
            Vertex& vertex = m_vertices[vA];
            m_edges[vertex.m_edgeStart + vertex.m_numEdges] = prevEdges[prevVertex.m_edgeStart + i];
            vertex.m_numEdges++;
        }
    }

    // Set Edge data of newly added edges.
    for (int i = 0; i < numNewEdges; i++)
    {
        const Cinfo::Connection& c = newEdges[i];
        int vA = std::min(c.m_vA, c.m_vB);
        int vB = std::max(c.m_vA, c.m_vB);

        Vertex& v = m_vertices[vA];
        Edge& e = m_edges[v.m_edgeStart + v.m_numEdges];
        e.m_otherVertex = vB;
        e.m_length = c.m_length > 0.f ? SimdFloat(c.m_length) : (m_positions[vA] - m_positions[vB]).length<3>();
        e.m_stiffness = SimdFloat(c.m_stiffness);
        v.m_numEdges++;
    }

    updateSolver();

    onParticlesAdded(newVertices);
}

void PointBasedSystem::createSolver(const Cinfo& cinfo)
{
    switch (cinfo.m_solverType)
    {
    case PointBasedSystemSolver::Type::POSITION_BASED_DYNAMICS:
    {
        m_solver = std::make_shared<PBD::Solver>(*this, cinfo.m_gravity, cinfo.m_solverIterations, cinfo.m_dampingFactor);
        break;
    }
    case PointBasedSystemSolver::Type::MASS_SPRING:
    {
        m_solver = std::make_shared<MassSpringSolver>(*this, cinfo.m_gravity, cinfo.m_dampingFactor);
        break;
    }
    default:
        assert(0);
        break;
    }
}

void PointBasedSystem::updateSolver()
{
    assert(m_solver);

    // [TODO] Implement proper update function for each solver instead of just recreating them.
    //        Creating a new PBD::Solver involves allocating all constraints again which can be costly.
    switch (m_solver->getType())
    {
    case PointBasedSystemSolver::Type::POSITION_BASED_DYNAMICS:
    {
        auto curSolver = static_cast<PBD::Solver*>(m_solver.get());
        m_solver = std::make_shared<PBD::Solver>(*this, curSolver->getGravity(), curSolver->getSolverIterations(), curSolver->getDampingFactor().getFloat());
        break;
    }
    case PointBasedSystemSolver::Type::MASS_SPRING:
    {
        auto curSolver = static_cast<MassSpringSolver*>(m_solver.get());
        m_solver = std::make_shared<MassSpringSolver>(*this, curSolver->getGravity(), curSolver->getDampingFactor().getFloat());
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

int PointBasedSystem::subscribeToOnParticleAdded(const OnParticleAddedFunc& f)
{
    int handle = m_onParticleAddedFuncs.size();
    while (m_onParticleAddedFuncs.find(handle) != m_onParticleAddedFuncs.end())
    {
        handle++;
    }
    m_onParticleAddedFuncs[handle] = f;
    return handle;
}

void PointBasedSystem::unsubscribeFromOnParticleAdded(int handle)
{
    auto itr = m_onParticleAddedFuncs.find(handle);
    if (itr != m_onParticleAddedFuncs.end())
    {
        m_onParticleAddedFuncs.erase(itr);
    }
}

void PointBasedSystem::onParticlesAdded(const Positions& posOfNewVertices) const
{
    // Fire callback functions
    if (posOfNewVertices.size() > 0)
    {
        for (const auto& itr : m_onParticleAddedFuncs)
        {
            itr.second(posOfNewVertices);
        }
    }
}
