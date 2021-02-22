/*
* DefaultGenomeSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <NEAT/GenomeSelectorBase.h>
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
        virtual bool setGenomes(const GenomeDatas& genomes) override;

        // Select a random genome.
        virtual auto selectGenome()->const GenomeData* override;

        // Select two random genomes.
        virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override;

        inline void setInterSpeciesCrossOverRate(float interSpeciesCrossOverRate) { m_interSpeciesCrossOverRate = interSpeciesCrossOverRate; }

        inline void skipStagnantSpecies(bool enable) { m_skipStagnantSpecies = enable; }

    protected:
        // Select a random genome between start and end (not including end) in m_genomes array.
        auto selectGenome(int start, int end)->const GenomeData*;

        // Returns SpeciesId of the given genome.
        SpeciesId getSpeciesId(const GenomeData& gd) const;

        // Returns true if the species of the given genome is reproducible.
        bool isGenomeReproducible(const GenomeData& gd) const;

        // Start and end index of genomes in m_genomes array for each species.
        struct IndexSet
        {
            int m_start, m_end;
        };

        const Generation* m_generation; // The generation.
        std::vector<const GenomeData*> m_genomes; // The genomes in the generation.
        std::vector<float> m_sumFitness; // Sum values of genomes' fitness.
        std::unordered_map<SpeciesId, IndexSet> m_spciecesStartEndIndices; // Intermediate data used internally.

        // Probability to select two genomes from different species when selectTwoGenomes() is called.
        float m_interSpeciesCrossOverRate = 0.001f;

        // Indicates whether to skip stagnant species during selection or not.
        bool m_skipStagnantSpecies = true;
    };
}
