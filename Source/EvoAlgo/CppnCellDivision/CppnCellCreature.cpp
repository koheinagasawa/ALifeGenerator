/*
* CppnCellCreature.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <EvoAlgo/EvoAlgo.h>
#include <EvoAlgo/CppnCellDivision/CppnCellCreature.h>

CppnCellCreature::CppnCellCreature(const Cinfo& cinfo)
    : m_simulation(cinfo.m_simulation)
    , m_genome(cinfo.m_genomeCinfo)
    , m_divisionInterval(cinfo.m_divisionInterval)
    , m_numMaxCells(cinfo.m_numMaxCells)
    , m_stiffness(cinfo.m_connectionStiffness)
{
    // Set 0 as generation index to initial cells.
    m_generationCounts.resize(m_simulation->getVertexPositions().size(), 0);
}

void CppnCellCreature::step(float deltaTime)
{
    // Check if we need to perform cell division.
    if (m_intervalCounter++ >= m_divisionInterval)
    {
        if ((int)m_simulation->getVertices().size() < m_numMaxCells)
        {
            divide();
        }

        m_intervalCounter = 0;
    }
}

void CppnCellCreature::divide()
{
    bool cellAdded = false;

    const PointBasedSystem::Vertices& vertices = m_simulation->getVertices();
    const PointBasedSystem::Edges& edges = m_simulation->getEdges();
    const int numCells = (int)vertices.size();

    // Helper struct to remember indices of an edge and the other vertex which the edge connects to.
    struct NeighborEdge
    {
        int m_otherVertex;
        int m_edgeIdx;
    };

    // Type definition of neighbors in one cell.
    using Neighbors = std::vector<NeighborEdge>;

    // Prepare neighbors for all the cells.
    std::vector<Neighbors> neighborsList;
    neighborsList.resize(numCells);

    // Create the arrays of neighbors.
    for (int vIdx = 0; vIdx < numCells; vIdx++)
    {
        const PointBasedSystem::Vertex& v = vertices[vIdx];
        neighborsList[vIdx].reserve(v.m_numEdges);
        for (int e = v.m_edgeStart, i = 0; i < v.m_numEdges; e++, i++)
        {
            // Remember as neighbor from both ways.
            int otherIdx = edges[e].m_otherVertex;
            neighborsList[vIdx].push_back({ otherIdx, e });
            neighborsList[otherIdx].push_back({ vIdx, e });
        }
    }

    std::vector<bool> isCellDivided;
    isCellDivided.resize(numCells, false);

    // Helper struct to store information of newly added cells.
    struct NewCell
    {
        Vector4 m_position;
        Vector4 m_direction;
        Vector4 m_origParentPos;
        int m_generation;
        int m_parentIdx;
        int m_cellIdx;
    };

    std::vector<NewCell> newCells;
    int newCellId = numCells;

    const PointBasedSystem::Positions prevPositions = m_simulation->getVertexPositions(); // Create a copy of original positions.

    // Try to divide all the existing cells.
    {
        using InputType = CppnCreatureGenome::InputNode;
        std::vector<float> inputNodeValues;
        inputNodeValues.resize((int)InputType::NUM_INPUT_NODES, 0.f);

        for (int cellIdx = 0; cellIdx < numCells; cellIdx++)
        {
            const Vector4& parentPos = prevPositions[cellIdx];
            const int generation = m_generationCounts[cellIdx] + 1;

            // Set input node values.
            {
                // Set parent position.
                inputNodeValues[(int)InputType::PARENT_POSITION_X] = parentPos.getComponent<0>().getFloat();
                inputNodeValues[(int)InputType::PARENT_POSITION_Y] = parentPos.getComponent<1>().getFloat();
                inputNodeValues[(int)InputType::PARENT_POSITION_Z] = parentPos.getComponent<2>().getFloat();

                const Neighbors& neighbors = neighborsList[cellIdx];
                const int numNeighbors = (int)neighbors.size();

                // Calculate average position of neighbor cells.
                Vector4 avgPos = Vec4_0;
                if (numNeighbors > 0)
                {
                    for (const NeighborEdge& ne : neighbors)
                    {
                        avgPos += prevPositions[ne.m_otherVertex];
                    }
                    avgPos /= SimdFloat((float)numNeighbors);
                }
                inputNodeValues[(int)InputType::NEIGHBOR_POSITION_X] = avgPos.getComponent<0>().getFloat();
                inputNodeValues[(int)InputType::NEIGHBOR_POSITION_Y] = avgPos.getComponent<1>().getFloat();
                inputNodeValues[(int)InputType::NEIGHBOR_POSITION_Z] = avgPos.getComponent<2>().getFloat();

                inputNodeValues[(int)InputType::NUM_NEIGHBORS] = (float)numNeighbors;
                inputNodeValues[(int)InputType::CELL_GENERATIONS] = (float)generation;
            }

            Vector4 direction;
            if (m_genome.evaluateDivision(inputNodeValues, direction))
            {
                // This cell should divide.
                const Vector4 offset = SimdFloat(m_simulation->getVertexRadius()) * direction;
                const Vector4 position = parentPos + offset;

                // Add a new cell.
                newCells.push_back(NewCell{ position, direction, parentPos, generation, cellIdx, newCellId++ });
                isCellDivided[cellIdx] = true;
                m_generationCounts.push_back(generation);

                // Update parent cell's position too.
                m_simulation->accessVertexPositions()[cellIdx] -= offset;
            }
        }
    }

    int numNewCells = (int)newCells.size();
    if (numNewCells > 0)
    {
        // Add divided cells to the PB system.

        const PointBasedSystem::Positions& curPositions = m_simulation->getVertexPositions();

        PointBasedSystem::Cinfo::Connections newConnections;
        std::vector<int> edgesToRemove;

        // Default edge length for the new cell.
        const float edgeLength = 2.0f * m_simulation->getVertexRadius();
        const SimdFloat distThresholdSq = SimdFloat(edgeLength * edgeLength * 1.73f);

        // Cinfo of new connections for newly divided cells.
        PointBasedSystem::Cinfo::Connection newConnectionCinfo;
        {
            newConnectionCinfo.m_stiffness = m_stiffness;
            newConnectionCinfo.m_length = edgeLength;
        }

        // Helper function to create a new connection.
        auto createNewConnection = [&newConnectionCinfo, &newConnections](int cellId1, int cellId2)
        {
            // Connect the new cell and the neighbor cell.
            newConnectionCinfo.m_vA = std::min(cellId1, cellId2);
            newConnectionCinfo.m_vB = std::max(cellId1, cellId2);
            newConnections.push_back(newConnectionCinfo);
        };

        // Create buffers to store positions and velocities for new cells.
        PointBasedSystem::Positions newPositions;
        newPositions.reserve(numNewCells);
        PointBasedSystem::Velocities newVelocities;
        newVelocities.reserve(numNewCells);

        const SimdFloat createNewCellNeighborEdgeThreshold(-0.1f);
        const SimdFloat removeParentNeighborEdgeThreshold(0.1f);

        // Iterate over newly added cells and add connections for them.
        for (const NewCell& newCell : newCells)
        {
            const int parentCellId = newCell.m_parentIdx;

            // Save its position and velocity.
            newPositions.push_back(newCell.m_position);
            newVelocities.push_back(m_simulation->getVertexVelocities()[parentCellId]); // Velocity is the same as its parent.

            // Add a new connection between the new cell and its parent.
            newConnectionCinfo.m_vA = parentCellId;
            newConnectionCinfo.m_vB = newCell.m_cellIdx;
            newConnections.push_back(newConnectionCinfo);

            // Add new connections between the new cell and neighbor cells of its parent.
            const std::vector<NeighborEdge>& neighbors = neighborsList[parentCellId];
            for (const NeighborEdge& neighbor : neighbors)
            {
                const int neighborCellId = neighbor.m_otherVertex;
                Vector4 prevParentToNeighbor = curPositions[neighborCellId] - newCell.m_origParentPos;
                prevParentToNeighbor.normalize<3>();
                const SimdFloat neighborDirDot = prevParentToNeighbor.dot<3>(newCell.m_direction);
                bool createNewCellNeighborEdge = neighborDirDot > createNewCellNeighborEdgeThreshold;
                bool removeParentNeighborEdge = neighborDirDot > removeParentNeighborEdgeThreshold;

                const SimdFloat distToNeighborSq = (curPositions[neighborCellId] - newCell.m_position).lengthSq<3>();

                if (isCellDivided[neighborCellId])
                {
                    // The neighbor cell was also divided.

                    // Find the divided cell from the neighbor cell.
                    const NewCell* otherNewCell = nullptr;
                    for (const NewCell& onc : newCells)
                    {
                        if (onc.m_parentIdx == neighborCellId)
                        {
                            otherNewCell = &onc;
                            break;
                        }
                    }

                    assert(otherNewCell);

                    const SimdFloat distToOtherNewCellSq = (otherNewCell->m_position - newCell.m_position).lengthSq<3>();

                    if (createNewCellNeighborEdge)
                    {
                        if (removeParentNeighborEdge)
                        {
                            // Remove the existing edge between the new cell's parent and the neighbor cell.
                            edgesToRemove.push_back(neighbor.m_edgeIdx);
                        }

                        if ((distToOtherNewCellSq < distThresholdSq) && (prevPositions[neighborCellId] - newCell.m_position).dot<3>(otherNewCell->m_direction) < SimdFloat_0)
                        {
                            // The two newly divided cells are closer than their parents.
                            if (neighborCellId > parentCellId)
                            {
                                // Add a new connection only when the parent of this new cell is smaller to avoid adding the connection twice.
                                createNewConnection(newCell.m_cellIdx, otherNewCell->m_cellIdx);
                            }
                        }
                        else if(distToNeighborSq < distThresholdSq)
                        {
                            // This new cell is close to the neighbor cell.
                            createNewConnection(newCell.m_cellIdx, neighbor.m_otherVertex);
                        }
                    }
                    else if ((neighborCellId > parentCellId) && (distToOtherNewCellSq < distThresholdSq) &&
                        ((newCell.m_origParentPos - otherNewCell->m_position).dot<3>(newCell.m_direction) < SimdFloat_0))
                    {
                        // The two newly divided cells are closer than their parents.
                        // Add a new connection only when the parent of this new cell is smaller to avoid adding the connection twice.
                        createNewConnection(newCell.m_cellIdx, otherNewCell->m_cellIdx);
                    }
                }
                else if(createNewCellNeighborEdge)
                {
                    if (distToNeighborSq < distThresholdSq)
                    {
                        // The neighbor cell was not divided and the neighbor cell is closer to the new cell than its parent.
                        // We connect the neighbor and the new cell and remove the existing edge between the neighbor and the parent if necessary.
                        createNewConnection(newCell.m_cellIdx, neighbor.m_otherVertex);
                    }

                    if (removeParentNeighborEdge)
                    {
                        // Remove the existing edge between the new cell's parent and the neighbor cell.
                        edgesToRemove.push_back(neighbor.m_edgeIdx);
                    }
                }
            }
        }

        // Sort edges to remove and remove duplicates.
        if (edgesToRemove.size() > 1)
        {
            std::sort(edgesToRemove.begin(), edgesToRemove.end());
            edgesToRemove.erase(std::unique(edgesToRemove.begin(), edgesToRemove.end()), edgesToRemove.end());
        }
        // Add/remove cells and edges
        m_simulation->addRemoveVerticesAndEdges(newPositions, newVelocities, newConnections, edgesToRemove);
    }
}