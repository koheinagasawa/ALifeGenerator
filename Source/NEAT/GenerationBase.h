/*
* GenerationBase.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#pragma once

#include <memory>
#include <vector>

#include <NEAT/GenomeBase.h>

class FitnessCalculatorBase
{
public:
    virtual float calcFitness(const GenomeBase& genome) const = 0;
};

class SpeciationDelegate
{
public:
};

DECLARE_ID(GenerationId);
DECLARE_ID(SpeciesId);
DECLARE_ID(GenomeId);

class GenerationBase
{
public:

    struct GenomeData
    {
    public:

        auto getGenome() const->const GenomeBase* { return m_genome.get(); }
        float getFitness() const { return m_fitness; }

    protected:

        std::shared_ptr<GenomeBase> m_genome;
        GenomeId m_id;
        float m_fitness = 0.f;
    };

    using GenomeDatas = std::vector<GenomeData>;
    using GenomeDatasPtr = std::shared_ptr<GenomeDatas>;

    virtual void createNewGeneration();

    inline int getNumGenomes() const { return m_numGenomes; }
    inline auto getFitnessCalculator() const->const FitnessCalculatorBase& { return *m_fitnessCalculator; }

    inline auto getId() const->GenerationId { return m_id; }

protected:

    std::shared_ptr<class MutationDelegate> m_mutationDelegate;
    std::shared_ptr<class CrossOverDelegate> m_crossOverDelegate;
    std::shared_ptr<class GenomeSelectorBase> m_genomeSelector;

    std::shared_ptr<FitnessCalculatorBase> m_fitnessCalculator;

    GenomeDatasPtr m_genomes;
    GenomeDatasPtr m_prevGenGenomes;
    int m_numGenomes;
    GenerationId m_id;
};

class MutationDelegate
{
public:
    using GenomeBasePtr = std::shared_ptr<GenomeBase>;
    using GenomeBasePtrs = std::vector<GenomeBasePtr>;

    // Structure to store information about newly added edges by mutate().
    struct MutationOut
    {
        struct NewEdgeInfo
        {
            NodeId m_sourceInNode;
            NodeId m_sourceOutNode;
            EdgeId m_newEdge;
        };

        void clear()
        {
            for (int i = 0; i < NUM_NEW_EDGES; i++)
            {
                m_newEdges[i].m_sourceInNode = NodeId::invalid();
                m_newEdges[i].m_sourceOutNode = NodeId::invalid();
                m_newEdges[i].m_newEdge = EdgeId::invalid();
            }

            m_numNodesAdded = 0;
            m_numEdgesAdded = 0;

            m_newNode = NodeId::invalid();
        }

        static constexpr int NUM_NEW_EDGES = 3;
        NewEdgeInfo m_newEdges[NUM_NEW_EDGES];
        NodeId m_newNode = NodeId::invalid();
        int m_numNodesAdded;
        int m_numEdgesAdded;
    };

    virtual void mutate(GenomeBasePtr genomeIn, MutationOut& mutationOut) = 0;

    virtual auto mutate(const GenerationBase::GenomeDatas& generation, int numGenomesToMutate, GenomeSelectorBase* genomeSelector)->GenomeBasePtrs = 0;
};

class CrossOverDelegate
{
public:
    using GenomeBasePtr = std::unique_ptr<GenomeBase>;
    using GenomeBasePtrs = std::vector<GenomeBasePtr>;

    virtual auto crossOver(const GenomeBase& genome1, const GenomeBase& genome2, bool sameFitness)->GenomeBasePtr = 0;

    virtual auto crossOver(const GenerationBase::GenomeDatas& generation, int numGenomesToCrossover, GenomeSelectorBase* genomeSelector)->GenomeBasePtrs = 0;
};

class GenomeSelectorBase
{
public:
    virtual void setGenomes(const GenerationBase::GenomeDatas& generation) = 0;
    virtual auto selectGenome()->GenerationBase::GenomeData* = 0;
    virtual void selectTwoGenomes(GenerationBase::GenomeData*& genome1, GenerationBase::GenomeData*& genome2) = 0;
};
