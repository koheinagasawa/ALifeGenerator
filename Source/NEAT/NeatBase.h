/*
* NeatBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GeneticAlgorithms/NEAT/Generation.h>

namespace NEAT
{
    class NeatBase
    {
    public:
        using GenerationPtr = Generation::GenerationPtr;

        void createNewGeneration();

        const Generation& getCurrentGeneration() const;

        void calcFitnessOfCurrentGen();
        float getMaxFitnessInCurrentGen() const;

    protected:
        virtual float calcFitness(const Genome& genome) const = 0;

        GenerationPtr m_currentGen;
        GenerationPtr m_previousGen;

        InnovationCounter m_innovationCounter;
    };
}
