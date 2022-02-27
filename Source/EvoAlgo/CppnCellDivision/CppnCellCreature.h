/*
* CppnCellCreature.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Physics/Systems/PointBasedSystem.h>
#include <EvoAlgo/GeneticAlgorithms/Base/GenomeBase.h>

// A class which represents multi-cellular organism.
// Cells are divided by CPPN genome.
// Cells are represented by particles connected each other and simulated by point based simulation such as PBD.
class CppnCellCreature : public System
{
public:
    // Type definition
    using PBSPtr = std::shared_ptr<PointBasedSystem>;
    using GenomePtr = std::shared_ptr<GenomeBase>;

    // Type of input nodes
    enum class InputNode : uint16_t
    {
        // Direction of parent cell : 3
        PARENT_POSITION_X = 0,
        PARENT_POSITION_Y,
        PARENT_POSITION_Z,

        //// Orientation of parent cell : 3
        //PARENT_ORIENTATION_X,
        //PARENT_ORIENTATION_Y,
        //PARENT_ORIENTATION_Z,

        // Averaged direction of neighbor cells : 3
        // Neighbor cells = cells connected to the parent by a single link
        NEIGHBOR_POSITION_X,
        NEIGHBOR_POSITION_Y,
        NEIGHBOR_POSITION_Z,

        //// Averaged orientation of neighbor cells : 3
        //NEIGHBOR_ORIENTATION_X,
        //NEIGHBOR_ORIENTATION_Y,
        //NEIGHBOR_ORIENTATION_Z,

        // Number of neighbor cells : 1
        NUM_NEIGHBORS,
        // Generation count of the new cell : 1
        CELL_GENERATIONS,

        // Total num of input nodes : 3 + 3 + 3 + 3 + 1 + 1 = 14  // 3 + 3 + 1 + 1 = 8
        NUM_INPUT_NODES
    };

    // Type of output nodes
    enum class OutputNode : uint16_t
    {
        // Divide? (boolean) : 1
        DIVIDE = 0,

        // Direction of divide : 3
        DIRECTION_X,
        DIRECTION_Y,
        DIRECTION_Z,

        //// Orientation : 3
        //ORIENTATION_X,
        //ORIENTATION_Y,
        //ORIENTATION_Z,

        // Total num of output nodes : 1 + 3 + 3 = 7  // 1 + 3 = 4
        NUM_OUTPUT_NODES
    };

    // Construction info
    struct Cinfo
    {
        // The point based system to run particle simulation.
        PBSPtr m_simulation;

        // The CPPN genome.
        GenomeBase* m_genome;

        // The maximum number of cells.
        int m_numMaxCells = 500;

        // Step intervals between cell division
        // [TODO] Consider to make it per cell parameter as an output of CPPN, or add some perturbation to prevent all the cells from dividing at once.
        int m_divisionInterval = 60;

        // Stiffness of cell connections.
        float m_connectionStiffness = 0.05f;
    };

    // Constructor.
    CppnCellCreature(const Cinfo& cinfo);

    // Step function. We perform cell divisions here if necessary.
    virtual void step(float deltaTime) override;

private:
    // Evaluate if a cell should divide or not.
    // Return true when the cell divides. Direction in which the new cell should be created is stored in 'direction'.
    // [TODO] Support orientation
    bool evaluateDivision(const std::vector<float>& inputNodeValues, Vector4& directionOut);

    // Divide cells.
    void divide();

    PBSPtr m_simulation;                    // Pointer to point based simulation.

    GenomeBase* m_genome;                   // The genomes.
    std::vector<int> m_generationCounts;    // Generation index of each cell.

    int m_divisionInterval;                 // Step interval between cell divisions.
    int m_intervalCounter;                  // Step interval counter.
    int m_numMaxCells;                      // The maximum number of cells.
    float m_stiffness;                      // Stiffness of cell connections.
};
