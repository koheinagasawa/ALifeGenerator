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

        // Cinfo of new connections for newly divided cells.
        PointBasedSystem::Cinfo::Connection newConnectionCinfo;
        {
            newConnectionCinfo.m_stiffness = m_stiffness;
            newConnectionCinfo.m_length = edgeLength;
        }

        // Helper function to connect the new cell and a neighbor cell of its parent.
        auto connectNewCellAndNeighbor = [&newConnectionCinfo, &newConnections, &edgesToRemove](const NewCell& newCell, const NeighborEdge& neighbor)
        {
            // Connect the new cell and the neighbor cell.
            newConnectionCinfo.m_vA = neighbor.m_otherVertex;
            newConnectionCinfo.m_vB = newCell.m_cellIdx;
            newConnections.push_back(newConnectionCinfo);
            // Remove the existing edge between the new cell's parent and the neighbor cell.
            edgesToRemove.push_back(neighbor.m_edgeIdx);
        };

        // Helper function to connect two newly divided cells.
        auto connectNewCells = [&newConnectionCinfo, &newConnections](const NewCell& newCell1, const NewCell& newCell2)
        {
            newConnectionCinfo.m_vA = std::min(newCell1.m_cellIdx, newCell2.m_cellIdx);
            newConnectionCinfo.m_vB = std::max(newCell1.m_cellIdx, newCell2.m_cellIdx);
            newConnections.push_back(newConnectionCinfo);
        };

        // Create buffers to store positions and velocities for new cells.
        PointBasedSystem::Positions newPositions;
        newPositions.reserve(numNewCells);
        PointBasedSystem::Velocities newVelocities;
        newVelocities.reserve(numNewCells);

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
                bool neighborIsCloserToDividedCell = (newCell.m_origParentPos - curPositions[neighborCellId]).dot<3>(newCell.m_direction) < SimdFloat_0;

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

                    if (neighborIsCloserToDividedCell)
                    {
                        if ((neighborCellId > parentCellId) &&
                            ((prevPositions[neighborCellId] - newCell.m_position).dot<3>(otherNewCell->m_direction) < SimdFloat_0))
                        {
                            // The two newly divided cells are closer than their parents.
                            // Add a new connection only when the parent of this new cell is smaller to avoid adding the connection twice.
                            connectNewCells(newCell, *otherNewCell);
                        }
                        else
                        {
                            // This new cell is close to the neighbor cell.
                            connectNewCellAndNeighbor(newCell, neighbor);
                        }
                    }
                    else if ((neighborCellId > parentCellId) &&
                        ((newCell.m_origParentPos - otherNewCell->m_position).dot<3>(newCell.m_direction) < SimdFloat_0))
                    {
                        // The two newly divided cells are closer than their parents.
                        // Add a new connection only when the parent of this new cell is smaller to avoid adding the connection twice.
                        connectNewCells(newCell, *otherNewCell);
                    }
                }
                else if(neighborIsCloserToDividedCell)
                {
                    // The neighbor cell was not divided.
                    // If the neighbor cell is closer to the new cell than its parent, then connect the neighbor and the new cell
                    // (and remove the existing edge between the neighbor and the parent).
                    connectNewCellAndNeighbor(newCell, neighbor);
                }
            }
        }

        // Add edges between new cells who share the same neighbor even though they are not directly neighbors each other.
        const SimdFloat distThresholdSq = SimdFloat(edgeLength * edgeLength);
        for (int i = 0; i < numNewCells; i++)
        {
            const NewCell& newCell1 = newCells[i];
            int parentCell1Id = newCell1.m_parentIdx;
            const std::vector<NeighborEdge>& neighbors = neighborsList[parentCell1Id];

            for (int j = i + 1; j < numNewCells; j++)
            {
                const NewCell& newCell2 = newCells[j];

                for (const NeighborEdge& neighbor : neighbors)
                {
                    if (neighbor.m_otherVertex == newCell2.m_parentIdx)
                    {
                        int vA = std::min(newCell1.m_cellIdx, newCell2.m_cellIdx);
                        int vB = std::max(newCell1.m_cellIdx, newCell2.m_cellIdx);

                        // Check if these cells already have an edge.
                        {
                            bool alreadyHasEdge = false;
                            for (const auto& c : newConnections)
                            {
                                if (c.m_vA == vA && c.m_vB == vB)
                                {
                                    alreadyHasEdge = true;
                                    break;
                                }
                            }

                            if (alreadyHasEdge)
                            {
                                break;
                            }
                        }

                        // Add a connection if the two new cells are close enough.
                        SimdFloat distanceSq = (newCell1.m_position - newCell2.m_position).lengthSq<3>();
                        if (distanceSq <= distThresholdSq)
                        {
                            newConnectionCinfo.m_vA = vA;
                            newConnectionCinfo.m_vB = vB;
                            newConnections.push_back(newConnectionCinfo);
                            break;
                        }
                    }
                }
            }
        }

        // Add/remove cells and edges
        std::sort(edgesToRemove.begin(), edgesToRemove.end());
        m_simulation->addRemoveVerticesAndEdges(newPositions, newVelocities, newConnections, edgesToRemove);
    }
}