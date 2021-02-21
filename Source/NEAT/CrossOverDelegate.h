/*
* CrossOverDelegate.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenomeGenerator.h>

class CrossOverDelegate : public GenomeGenerator
{
public:
    virtual auto crossOver(const GenomeBase& genome1, const GenomeBase& genome2, bool sameFitness)->GenomeBasePtr = 0;
};
