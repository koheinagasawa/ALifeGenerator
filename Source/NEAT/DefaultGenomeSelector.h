/*
* DefaultGenomeSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenerationBase.h>
#include <NEAT/Generation.h>

namespace NEAT
{
    class DefaultGenomeSelector : public GenomeSelectorBase
    {
    public:
        DefaultGenomeSelector(const Generation* generation, PseudoRandom& random);

        virtual bool setGenomes(const GenomeDatas& generation) override;
        virtual auto selectGenome()->const GenomeData* override;
        virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override;

        void setInterSpeciesCrossOverRate(float interSpeciesCrossOverRate);

    protected:

        auto selectGenome(int start, int end)->const GenomeData*;

        SpeciesId getSpeciesId(const GenomeData& gd) const;

        bool isGenomeReproducible(const GenomeData& gd) const;

        struct IndexSet
        {
            int m_start, m_end;
        };

        const Generation* m_generation;
        float m_interSpeciesCrossOverRate = 0.001f;

        std::vector<const GenomeData*> m_genomes;
        std::vector<float> m_sumFitness;
        std::unordered_map<SpeciesId, IndexSet> m_spciecesStartEndIndices;
    };
}
