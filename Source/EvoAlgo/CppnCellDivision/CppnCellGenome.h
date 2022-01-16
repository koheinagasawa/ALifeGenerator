/*
* CppnCellGenome.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <EvoAlgo/GeneticAlgorithms/Base/GenomeBase.h>
#include <EvoAlgo/GeneticAlgorithms/Base/Activations/ActivationProvider.h>
#include <Common/Math/Vector4.h>

// Genome class for CPPN creature.
// Every cell of a CPPN creature has this genome and use it for cell division.
class CppnCreatureGenome : public GenomeBase
{
public:
    //
    // Type Declarations
    //

    using Node = GenomeBase::Node;
    using Edge = GenomeBase::Edge;
    using ActivationPtr = std::shared_ptr<Activation>;

    // Struct used to construct a initial genome.
    struct Cinfo
    {
        int m_numInitialHiddenLayers = 2;

        std::vector<int> m_numNodeInInitialHiddenLayers;

        // Default value of bias node.
        float m_biasNodeValue = 1.0f;

        // Activation provider for the initial network.
        // If it's nullptr, input values are merely passed as an output of the node.
        const ActivationProvider* m_activationProvider = nullptr;

        bool m_randomizeInitialEdges = true;

        RandomGenerator* m_randomWeightsGenerator = nullptr;
        float m_minWeight = -5.0f;
        float m_maxWeight = 5.0f;
    };

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
        DEVIDE = 0,

        // Direction : 3
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

    // Constructor
    CppnCreatureGenome(const Cinfo& cinfo);

    // Evaluate if a cell owning this genome should divide or not.
    // Return true when the cell divides. Direction in which the new cell should be created is stored in 'direction'.
    // [TODO] Support orientation
    bool evaluateDivision(const std::vector<float>& inputNodeValues, Vector4& directionOut);
};