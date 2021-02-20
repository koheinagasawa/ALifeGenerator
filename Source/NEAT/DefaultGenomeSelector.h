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
    // Helper class to select a random genome by taking fitness into account.
    class DefaultGenomeSelector : public GenomeSelectorBase
    {
    public:
        // Constructor
        DefaultGenomeSelector(const Generation* generation, PseudoRandom& random);

        // Set genomes to select and initialize internal data.
        virtual bool setGenomes(const GenomeDatas& generation) override;

        // Select a random genome.
        virtual auto selectGenome()->const GenomeData* override;

        // Select two random genomes.
        virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override;

        inline void setInterSpeciesCrossOverRate(float interSpeciesCrossOverRate) { m_interSpeciesCrossOverRate = interSpeciesCrossOverRate; }

        inline void skipStagnantSpecies(bool enable) { m_skipStagnantSpecies = enable; }

    protected:

        // Select a random genome between start and end (not including end) in m_genomes array.
        auto selectGenome(int start, int end)->const GenomeData*;

        SpeciesId getSpeciesId(const GenomeData& gd) const;

        bool isGenomeReproducible(const GenomeData& gd) const;

        struct IndexSet
        {
            int m_start, m_end;
        };

        const Generation* m_generation;

        std::vector<const GenomeData*> m_genomes;
        std::vector<float> m_sumFitness;
        std::unordered_map<SpeciesId, IndexSet> m_spciecesStartEndIndices;

        float m_interSpeciesCrossOverRate = 0.001f;
        bool m_skipStagnantSpecies = true;
    };
}
