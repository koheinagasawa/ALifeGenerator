/*
* SpeciesBasedGenomeSelector.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/PseudoRandom.h>
#include <NEAT/GeneticAlgorithms/Base/Selectors/GenomeSelector.h>
#include <NEAT/GeneticAlgorithms/NEAT/Species.h>

namespace NEAT
{
    // Helper class to select a random genome by taking fitness into account.
    class SpeciesBasedGenomeSelector : public GenomeSelector
    {
    public:
        // Type declarations.
        using SpeciesPtr = std::shared_ptr<Species>;
        using SpeciesList = std::unordered_map<SpeciesId, SpeciesPtr>;
        using GenomeSpeciesMap = std::unordered_map<GenomeId, SpeciesId>;
        using GenomeDataPtrs = std::vector<const GenomeData*>;

        // Constructor
        SpeciesBasedGenomeSelector(const GenomeDatas& genomes, const SpeciesList& species, const GenomeSpeciesMap& genomeSpeciesMap, PseudoRandom* random = nullptr);

        // Select a random genome.
        virtual auto selectGenome()->const GenomeData* override;

        // Select two random genomes.
        virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override;

        // Returns the number of genomes which could be selected by this selector.
        inline int getNumGenomes() const { return m_numGenomes; }

        void setNumGenomesToSelect(int numGenomes);

        inline bool hasSpeciesMoreThanOneMember() const { return m_hasSpeciesMoreThanOneMember; }

    protected:
        // Select a random genome between start and end (not including end) in m_genomes array.
        auto selectGenome(int start, int end)->const GenomeData*;

        struct SpeciesData
        {
            int m_population = 0;
            int m_remainingPopulation = 0;
            float m_sumFitness = 0;
            GenomeDataPtrs m_genomes;
            SpeciesPtr m_species;
        };

        std::vector<SpeciesData> m_speciesData;
        int m_currentSpeciesDataIndex;
        float m_totalFitness;
        int m_numGenomes;

        // Indicates whether to skip stagnant species during selection or not.
        bool m_skipStagnantSpecies = true;

        bool m_hasSpeciesMoreThanOneMember;

        // Random generator.
        PseudoRandom& m_random;
    };
}
