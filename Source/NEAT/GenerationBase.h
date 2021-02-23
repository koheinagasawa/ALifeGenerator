/*
* GenerationBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/PseudoRandom.h>
#include <NEAT/GenomeBase.h>
#include <NEAT/GenomeGenerator.h>

DECLARE_ID(GenerationId);
DECLARE_ID(SpeciesId);
DECLARE_ID(GenomeId);

// Base class to calculate fitness of a genome.
class FitnessCalculatorBase
{
public:
    virtual float calcFitness(const GenomeBase& genome) const = 0;
};

// Base class of generation used for generic algorithms.
class GenerationBase
{
public:
    // Type declarations.
    using GenomeBasePtr = std::shared_ptr<GenomeBase>;
    using CGenomeBasePtr = std::shared_ptr<const GenomeBase>;
    using FitnessCalcPtr = std::shared_ptr<FitnessCalculatorBase>;
    using GeneratorPtr = std::shared_ptr<GenomeGenerator>;
    using GeneratorPtrs = std::vector<GeneratorPtr>;

    // Struct holding a genome and its fitness.
    struct GenomeData
    {
    public:
        // Default constructor
        GenomeData() = default;

        // Constructor with a pointer to the genome and its id.
        GenomeData(GenomeBasePtr genome, GenomeId id);

        // Initialize by a pointer to the genome and its id.
        void init(GenomeBasePtr genome, GenomeId id);

        inline GenomeId getId() const { return m_id; }
        inline auto getGenome() const->const CGenomeBasePtr { return m_genome; }
        inline float getFitness() const { return m_fitness; }
        inline void setFitness(float fitness) { m_fitness = fitness; }

    protected:

        GenomeBasePtr m_genome; // The genome.
        float m_fitness = 0.f;  // Genome's fitness.
        GenomeId m_id;
    };

    // Type declarations.
    using GenomeDatas = std::vector<GenomeData>;
    using GenomeDatasPtr = std::shared_ptr<GenomeDatas>;

    // Destructor to make it abstract class.
    virtual ~GenerationBase() = 0 {}

    // Proceed and evolve this generation into a new generation.
    // New set of genomes will be generated from the current set of genomes. GenerationId will be incremented.
    virtual void evolveGeneration();

    // Calculate fitness of all the genomes.
    void calcFitness();

    inline int getNumGenomes() const { return m_numGenomes; }
    inline auto getFitnessCalculator() const->const FitnessCalculatorBase& { return *m_fitnessCalculator; }
    inline auto getId() const->GenerationId { return m_id; }

protected:
    // Type declarations.
    using GenomeSelectorPtr = std::shared_ptr<class GenomeSelector>;
    using FitnessCalculatorPtr = std::shared_ptr<FitnessCalculatorBase>;

    // Constructor
    GenerationBase(GenerationId id, int numGenomes, FitnessCalcPtr fitnessCalc, RandomGenerator* randomGenerator);

    // Called before/after generation of genomes inside evolveGeneration().
    virtual void preUpdateGeneration() {}
    virtual void postUpdateGeneration() {}

    // Returns GenomeSelector. This GenomeSelector is used to pass GenomeGenerators in when we evolve a new generation.
    virtual auto createSelector()->GenomeSelectorPtr = 0;

    // Called inside createNewGeneration().
    void addGenome(GenomeBasePtr genome);

    GeneratorPtrs m_generators;                 // Genome generators used to evolve generation.
    FitnessCalculatorPtr m_fitnessCalculator;   // The fitness calculator.
    GenomeDatasPtr m_genomes;                   // Genomes in the current generation.
    GenomeDatasPtr m_prevGenGenomes;            // Genomes in the previous generation.
    RandomGenerator* m_randomGenerator = nullptr;
    int m_numGenomes;
    GenerationId m_id;
};
