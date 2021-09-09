/*
* GenomeModifier.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <EvoAlgo/GeneticAlgorithms/Base/GenomeBase.h>

// Base class which modifies genomes.
class GenomeModifier
{
public:
    // Type declarations
    using GenomeBasePtr = std::shared_ptr<GenomeBase>;

    // Modifies the genomes
    virtual void modifyGenomes(GenomeBasePtr& genome) = 0;
};
