/*
* GenerationBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <memory>
#include <vector>

#include <Common/PseudoRandom.h>
#include <NEAT/GenomeBase.h>

DECLARE_ID(GenerationId);
DECLARE_ID(SpeciesId);
DECLARE_ID(GenomeId);

class FitnessCalculatorBase
{
public:
    virtual float calcFitness(const GenomeBase& genome) const = 0;
};

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

class GenerationBase
{
public:
    using GenomeBasePtr = std::shared_ptr<GenomeBase>;
    using FitnessCalcPtr = std::shared_ptr<FitnessCalculatorBase>;
    using GeneratorPtr = std::shared_ptr<GenomeGenerator>;
    using GeneratorPtrs = std::vector<GeneratorPtr>;

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
        inline auto getGenome() const->const GenomeBase* { return m_genome.get(); }
        inline float getFitness() const { return m_fitness; }
        inline void setFitness(float fitness) { m_fitness = fitness; }

    protected:

        std::shared_ptr<GenomeBase> m_genome;
        GenomeId m_id;
        float m_fitness = 0.f;
    };

    using GenomeDatas = std::vector<GenomeData>;
    using GenomeDatasPtr = std::shared_ptr<GenomeDatas>;

    virtual void createNewGeneration();

    // Calculate fitness of all the genomes.
    void calcFitness();

    inline int getNumGenomes() const { return m_numGenomes; }
    inline auto getFitnessCalculator() const->const FitnessCalculatorBase& { return *m_fitnessCalculator; }

    inline auto getId() const->GenerationId { return m_id; }

protected:
    using GenomeSelectorPtr = std::shared_ptr<class GenomeSelectorBase>;

    GenerationBase(GenerationId id, int numGenomes, FitnessCalcPtr fitnessCalc);

    virtual void preUpdateGeneration() = 0;
    virtual void postUpdateGeneration() = 0;

    virtual auto createSelector()->GenomeSelectorPtr = 0;

    // Called inside createNewGeneration().
    void addGenome(GenomeBasePtr genome);

    GeneratorPtrs m_generators;

    std::shared_ptr<FitnessCalculatorBase> m_fitnessCalculator;

    GenomeDatasPtr m_genomes;
    GenomeDatasPtr m_prevGenGenomes;
    PseudoRandom* m_randomGenerator = nullptr;
    int m_numGenomes;
    GenerationId m_id;
};

class MutationDelegate : public GenomeGenerator
{
public:
    // Structure to store information about newly added edges by mutate().
    struct MutationOut
    {
        struct NewEdgeInfo
        {
            NodeId m_sourceInNode;
            NodeId m_sourceOutNode;
            EdgeId m_newEdge;
        };

        void clear();

        static constexpr int NUM_NEW_EDGES = 3;
        NewEdgeInfo m_newEdges[NUM_NEW_EDGES];
        NodeId m_newNode = NodeId::invalid();
        int m_numNodesAdded;
        int m_numEdgesAdded;
    };

    virtual void mutate(GenomeBasePtr genomeIn, MutationOut& mutationOut) = 0;
};

class CrossOverDelegate : public GenomeGenerator
{
public:
    virtual auto crossOver(const GenomeBase& genome1, const GenomeBase& genome2, bool sameFitness)->GenomeBasePtr = 0;
};

// Abstract class to select a genome.
class GenomeSelectorBase
{
public:
    using GenomeDatas = GenerationBase::GenomeDatas;
    using GenomeData = GenerationBase::GenomeData;

    // Constructor
    GenomeSelectorBase(PseudoRandom& random) : m_random(random) {}

    // Set genomes to select and initialize internal data.
    virtual bool setGenomes(const GenomeDatas& generation) = 0;

    // Select a random genome.
    virtual auto selectGenome()->const GenomeData* = 0;

    // Select two random genomes.
    virtual void selectTwoGenomes(const GenomeData*& genome1, const GenomeData*& genome2) = 0;

protected:
    PseudoRandom& m_random;
};
