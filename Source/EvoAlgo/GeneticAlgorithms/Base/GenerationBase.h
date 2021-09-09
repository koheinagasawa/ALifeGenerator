/*
* GenerationBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <Common/PseudoRandom.h>
#include <EvoAlgo/GeneticAlgorithms/Base/GenomeBase.h>
#include <EvoAlgo/NeuralNetwork/NeuralNetworkEvaluator.h>

DECLARE_ID(GenerationId);
DECLARE_ID(GenomeId);

// Base class to calculate fitness of a genome.
class FitnessCalculatorBase
{
public:
    using FitnessCalcPtr = std::shared_ptr<FitnessCalculatorBase>;

    // This function has to be implemented to calculate fitness of a genome.
    virtual float calcFitness(GenomeBase* genome) = 0;

    // Return a clone of this calculator.
    virtual FitnessCalcPtr clone() const = 0;

protected:
    // This function can be called inside calcFitness() to evaluate a genome to assess its fitness.
    void evaluateGenome(GenomeBase* genome, const std::vector<float>& inputNodeValues, float biasNodeValue = 0.f);

public:
    NeuralNetworkEvaluator m_evaluator;
};

// Base class of generation used for generic algorithms.
class GenerationBase
{
public:
    // Type declarations.
    using GenomeBasePtr = std::shared_ptr<GenomeBase>;
    using CGenomeBasePtr = std::shared_ptr<const GenomeBase>;
    using FitnessCalcPtr = std::shared_ptr<FitnessCalculatorBase>;
    using FitnessCalculators = std::vector<FitnessCalcPtr>;
    using GeneratorPtr = std::shared_ptr<class GenomeGenerator>;
    using GeneratorPtrs = std::vector<GeneratorPtr>;
    using ModifierPtr = std::shared_ptr<class GenomeModifier>;
    using ModifierPtrs = std::vector<ModifierPtr>;

    // Struct holding a genome and its fitness.
    struct GenomeData
    {
    public:
        // Default constructor
        GenomeData() = default;

        // Constructor with a pointer to the genome and its id.
        GenomeData(GenomeBasePtr genome, GenomeId id);

        // Initialize by a pointer to the genome and its id.
        void init(GenomeBasePtr genome, bool isProtected, GenomeId id);

        inline GenomeId getId() const { return m_id; }
        inline auto getGenome() const->const CGenomeBasePtr { return m_genome; }
        inline float getFitness() const { return m_fitness; }
        inline void setFitness(float fitness) { m_fitness = fitness; }
        inline bool isProtected() const { return m_isProtected; }
        inline void setProtected(bool protect) { m_isProtected = protect; }

    protected:
        GenomeBasePtr m_genome;         // The genome.
        float m_fitness = 0.f;          // Genome's fitness.
        bool m_isProtected = false;     // Protected genome will not be modified and remain untouched in the next generation.
        GenomeId m_id;

        friend class GenerationBase;
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

    // Return the number of genomes.
    inline int getNumGenomes() const { return m_numGenomes; }

    // Return fitness calculators.
    inline auto getFitnessCalculators() const->const FitnessCalculators& { return m_fitnessCalculators; }

    // Return generation id.
    inline auto getId() const->GenerationId { return m_id; }

    // Return genome data.
    inline auto getGenomeData() const->const GenomeDatas& { return *m_genomes; }

protected:
    // Type declarations.
    using GenomeSelectorPtr = std::shared_ptr<class GenomeSelector>;
    using FitnessCalculatorPtr = std::shared_ptr<FitnessCalculatorBase>;

    // Constructor
    GenerationBase(GenerationId id, int numGenomes, RandomGenerator* randomGenerator);

    // Create fitness calculators for each thread by copying fitnessCalc.
    void createFitnessCalculators(FitnessCalcPtr fitnessCalc, int numThreads);

    // Called before/after generation of genomes inside evolveGeneration().
    virtual void preUpdateGeneration() {}
    virtual void postUpdateGeneration() {}

    // Returns GenomeSelector. This GenomeSelector is used to pass GenomeGenerators in when we evolve a new generation.
    virtual auto createSelector()->GenomeSelectorPtr = 0;

    GeneratorPtrs m_generators;                     // Genome generators used to evolve generation.
    ModifierPtrs m_modifiers;                       // Genome modifiers used to evolve generation.
    FitnessCalculators m_fitnessCalculators;        // The fitness calculator. There is a one calculator per thread.
    GenomeDatasPtr m_genomes;                       // Genomes in the current generation.
    GenomeDatasPtr m_prevGenGenomes;                // Genomes in the previous generation.
    RandomGenerator* m_randomGenerator = nullptr;   // Random generator.
    int m_numGenomes;                               // The number of genomes.
    float m_bestFitness = 0;                        // The best fitness in this generation.
    GenerationId m_id;                              // Generation id incremented at every evolveGeneration() call.
};
