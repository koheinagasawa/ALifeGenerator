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

        // This function should be called before the first selection.
        virtual void preSelection(int numGenomesToSelect, SelectionMode mode) override;

        // This function should be called after the last selection.
        virtual void postSelection() override;

        // Select a random genome.
        virtual auto selectGenome()->const GenomeData* override;

        // Set the probability to select two genomes from different species when selectTwoGenomes() is called.
        inline void setInterSpeciesSelectionRate(float interSpeciesCrossOverRate) { m_interSpeciesSelectionRate = interSpeciesCrossOverRate; }

        // Select two random genomes.
        virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override;

        // Returns the number of genomes which could be selected by this selector.
        inline int getNumGenomes() const { return m_numGenomes; }

        void setSpeciesPopulations(int numGenomesToSelect);

        inline bool hasSpeciesMoreThanOneMember() const { return m_hasSpeciesMoreThanOneMember; }

    protected:
        struct SpeciesData
        {
            SpeciesData() = default;
            SpeciesData(SpeciesPtr species);

            inline float getSumFitness() const { return m_cumulativeFitnesses.back(); }
            inline int getNumGenomes() const { return (int)m_genomes.size(); }

            int m_population = 0; // Distributed population of this species.
            int m_remainingPopulation = 0; // Remaining of the distributed population.
            GenomeDataPtrs m_genomes; // Members of the species.
            SpeciesPtr m_species; // The species.
            std::vector<float> m_cumulativeFitnesses; // Cumulative sum of fitness of the members.
        };

        auto selectGenomeImpl()->const GenomeData*;

        void decrementPopulationOfCurrentSpecies();

        std::vector<SpeciesData> m_speciesData;
        int m_currentSpeciesDataIndex = -1;
        float m_totalFitness = 0.f;
        int m_numGenomes = 0;

        // Indicates whether to skip stagnant species during selection or not.
        bool m_skipStagnantSpecies = true;

        bool m_hasSpeciesMoreThanOneMember = false;

        // Probability to select two genomes from different species when selectTwoGenomes() is called.
        float m_interSpeciesSelectionRate = 0.001f;
        int m_numInterSpeciesSelection = 0;
        std::vector<float> m_cumulativeSpeciesFitness;

        SelectionMode m_mode;

        // Random generator.
        PseudoRandom& m_random;
    };
}
