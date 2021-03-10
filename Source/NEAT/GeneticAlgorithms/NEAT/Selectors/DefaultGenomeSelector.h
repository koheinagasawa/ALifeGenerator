/*
* DefaultGenomeSelector.h
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
    class DefaultGenomeSelector : public GenomeSelector
    {
    public:
        // Type declarations.
        using SpeciesPtr = std::shared_ptr<Species>;
        using SpeciesList = std::unordered_map<SpeciesId, SpeciesPtr>;
        using GenomeSpeciesMap = std::unordered_map<GenomeId, SpeciesId>;
        using GenomeDataPtrs = std::vector<const GenomeData*>;

        // Constructor
        DefaultGenomeSelector(const GenomeDatas& genomes, const SpeciesList& species, const GenomeSpeciesMap& genomeSpeciesMap, PseudoRandom* random = nullptr);

        // Select a random genome.
        virtual auto selectGenome()->const GenomeData* override;

        // Select two random genomes.
        virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) override;

        // Returns the number of genomes which could be selected by this selector.
        inline int getNumGenomes() const { return (int)m_genomes.size(); }

        void setNumGenomesToSelect();

        bool hasSpeciesMoreThanOneMember() const;

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

        GenomeDataPtrs m_genomes; // The genomes in the generation.
        std::vector<float> m_sumFitness; // Sum values of genomes' fitness.
        std::unordered_map<SpeciesId, IndexSet> m_spciecesStartEndIndices; // Intermediate data used internally.
        const SpeciesList& m_species;
        const GenomeSpeciesMap& m_genomeSpeciesMap;

        // Indicates whether to skip stagnant species during selection or not.
        bool m_skipStagnantSpecies = true;

        // Random generator.
        PseudoRandom& m_random;
    };
}
