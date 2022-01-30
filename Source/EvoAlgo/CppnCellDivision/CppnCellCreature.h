/*
* CppnCellCreature.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Physics/Systems/PointBasedSystem.h>
#include <EvoAlgo/CppnCellDivision/CppnCellGenome.h>

// A class which represents multi-cellular organism.
// Cells are divided by CPPN genome.
// Cells are represented by particles connected each other and simulated by point based simulation such as PBD.
class CppnCellCreature : public System
{
public:
    // Type definition
    using PBSPtr = std::shared_ptr<PointBasedSystem>;

    // Construction info
    struct Cinfo
    {
        // The point based system to run particle simulation.
        PBSPtr m_simulation;

        // Construction info of CPPN genome.
        CppnCreatureGenome::Cinfo m_genomeCinfo;

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

    // Divide cells.
    void divide();

    PBSPtr m_simulation;                    // Pointer to point based simulation.

    CppnCreatureGenome m_genome;            // The genomes.
    std::vector<int> m_generationCounts;    // Generation index of each cell.

    int m_divisionInterval;                 // Step interval between cell divisions.
    int m_intervalCounter;                  // Step interval counter.
    int m_numMaxCells;                      // The maximum number of cells.
    float m_stiffness;                      // Stiffness of cell connections.
};
