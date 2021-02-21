/*
* GenomeGenerator.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenomeBase.h>

class GenomeGenerator
{
public:
    using GenomeBasePtr = std::shared_ptr<GenomeBase>;
    using GenomeBasePtrs = std::vector<GenomeBasePtr>;

    virtual void generate(int numTotalGenomes, int numRemaningGenomes, class GenomeSelectorBase* genomeSelector) = 0;

    inline int getNumGeneneratedGenomes() const { return (int)m_generatedGenomes.size(); }

    inline auto getGeneratedGenomes() const->const GenomeBasePtrs { return m_generatedGenomes; }

protected:
    GenomeBasePtrs m_generatedGenomes;
};
