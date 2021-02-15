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

class MutationDelegate
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

    virtual void mutate(GenomeBase* genomeIn, MutationOut& mutationOut) = 0;
};

class CrossOverDelegate
{
public:

    virtual auto crossOver(const GenomeBase& genome1, const GenomeBase& genome2, bool sameFitness)->GenomeBase* = 0;
};

class SpeciationDelegate
{
public:
};

class GenomeSelectorBase
{
public:

    virtual auto selectGenome()->GenomeBase* = 0;
    virtual void selectTwoGenomes(GenomeBase*& genome1, GenomeBase*& genome2) = 0;
};

DECLARE_ID(GenerationId);
DECLARE_ID(SpeciesId);
DECLARE_ID(GenomeId);

class GenerationBase
{
public:
    virtual void createNewGeneration();

    inline int getNumGenomes() const { return m_numGenomes; }
    inline auto getFitnessCalculator() const->const FitnessCalculatorBase& { return *m_fitnessCalculator; }

    inline auto getId() const->GenerationId { return m_id; }

protected:

    std::shared_ptr<MutationDelegate> m_mutationDelegate;
    std::shared_ptr<CrossOverDelegate> m_crossOverDelegate;
    std::shared_ptr<SpeciationDelegate> m_speciationDelegate;
    std::shared_ptr<GenomeSelectorBase> m_genomeSelector;

    std::shared_ptr<FitnessCalculatorBase> m_fitnessCalculator;

    int m_numGenomes;
    GenerationId m_id;
};

