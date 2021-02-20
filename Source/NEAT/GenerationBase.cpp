/*
* GenerationBase.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GenerationBase.h>

GenerationBase::GenomeData::GenomeData(GenomeBasePtr genome, GenomeId id)
    : m_genome(genome)
    , m_id(id)
{
}

void GenerationBase::GenomeData::init(GenomeBasePtr genome, GenomeId id)
{
    m_genome = genome;
    m_id = id;
    m_fitness = 0.f;
}

GenerationBase::GenerationBase(GenerationId id, int numGenomes, FitnessCalcPtr fitnessCalc)
    : m_fitnessCalculator(fitnessCalc)
    , m_numGenomes(numGenomes)
    , m_id(id)
{
    assert(m_numGenomes > 0);
    assert(m_fitnessCalculator);
}

void GenerationBase::createNewGeneration()
{
    // TODO: Break this function into smaller parts so that we can test it more thoroughly.
    // TODO: Profile each process by adding timers.

    const int numGenomes = getNumGenomes();
    assert(numGenomes > 1);

    std::swap(m_genomes, m_prevGenGenomes);

    m_numGenomes = 0;

    // Allocate buffer of GenomeData if it's not there yet.
    if (!m_genomes)
    {
        m_genomes = std::make_shared<GenomeDatas>();
    }
    if (m_genomes->size() != numGenomes)
    {
        m_genomes->resize(numGenomes);
    }

    preUpdateGeneration();

    int numGenomesToAdd = numGenomes - m_numGenomes;

    GenomeSelectorPtr selector = createSelector();

    using NewGenomePtrsOut = std::vector<GenomeBasePtr>;

    // Select and mutate genomes.
    {
        const int numGenomesToMutate = std::min(numGenomesToAdd, int(numGenomes * (1.f - m_params.m_crossOverRate)));
        NewGenomePtrsOut newGenomes = m_mutationDelegate->mutate(numGenomesToMutate, selector.get());

        for (auto& newGenome : newGenomes)
        {
            addGenome(newGenome);
            numGenomesToAdd--;
        }
    }

    // Select and generate new genomes by crossover.
    {
        const int numGenomesToCrossover = numGenomesToAdd;
        NewGenomePtrsOut newGenomes = m_crossOverDelegate->crossOver(numGenomesToCrossover, selector.get());

        for (auto& newGenome : newGenomes)
        {
            addGenome(newGenome);
            numGenomesToAdd--;
        }
    }

    // We should have added all the genomes at this point.
    assert(m_genomes->size() == m_prevGenGenomes->size());

    // Evaluate all genomes.
    calcFitness();

    postUpdateGeneration();

    // Update the generation id.
    m_id = GenerationId(m_id.val() + 1);
}

void GenerationBase::calcFitness()
{
    for (GenomeData& gd : *m_genomes)
    {
        gd.setFitness(m_fitnessCalculator->calcFitness(*gd.getGenome()));
    }
}

void GenerationBase::addGenome(GenomeBasePtr genome)
{
    (*m_genomes)[m_numGenomes].init(genome, GenomeId(m_numGenomes));
    m_numGenomes++;
}

void MutationDelegate::MutationOut::clear()
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
