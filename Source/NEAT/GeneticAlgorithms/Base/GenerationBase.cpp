/*
* GenerationBase.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/Base/GenerationBase.h>
#include <NEAT/GeneticAlgorithms/Base/Generators/GenomeGenerator.h>
#include <NEAT/GeneticAlgorithms/Base/Modifiers/GenomeModifier.h>

GenerationBase::GenomeData::GenomeData(GenomeBasePtr genome, GenomeId id)
    : m_genome(genome)
    , m_id(id)
{
}

void GenerationBase::GenomeData::init(GenomeBasePtr genome, bool isProtected, GenomeId id)
{
    m_genome = genome;
    m_fitness = 0.f;
    m_isProtected = isProtected;
    m_id = id;
}

GenerationBase::GenerationBase(GenerationId id, int numGenomes, FitnessCalcPtr fitnessCalc, RandomGenerator* randomGenerator)
    : m_fitnessCalculator(fitnessCalc)
    , m_randomGenerator(randomGenerator)
    , m_numGenomes(numGenomes)
    , m_id(id)
{
    assert(m_numGenomes > 0);
    assert(m_fitnessCalculator);
}

void GenerationBase::evolveGeneration()
{
    // TODO: Profile each process by adding timers.
    assert(m_genomes);
    assert(m_generators.size() > 0);
    const int numGenomes = getNumGenomes();
    assert(numGenomes > 1);

    preUpdateGeneration();

    // Create a genome selector
    GenomeSelectorPtr selector = createSelector();

    std::swap(m_genomes, m_prevGenGenomes);

    // Allocate buffer of GenomeData if it's not there yet.
    if (!m_genomes)
    {
        m_genomes = std::make_shared<GenomeDatas>();
    }
    if (m_genomes->size() != numGenomes)
    {
        m_genomes->resize(numGenomes);
    }

    int numGenomesToAdd = numGenomes;
    m_numGenomes = 0;

    // Create genomes for new generations by applying each genome generators.
    for (GeneratorPtr& generator : m_generators)
    {
        // [todo] Add a way to notify generator that if it's the last generator in this generation
        //        so that it can generate all the remaining genomes.
        generator->generate(numGenomes, numGenomesToAdd, selector.get());

        const bool protecteGenomes = generator->shouldGenomesProtected();
        for (auto& newGenome : generator->getGeneratedGenomes())
        {
            addGenome(newGenome, protecteGenomes);
        }

        numGenomesToAdd -= generator->getNumGeneratedGenomes();
    }

    // We should have added all the genomes at this point.
    assert(m_numGenomes == m_prevGenGenomes->size());

    // Modify genomes
    for (GenomeData& genomeData : *m_genomes)
    {
        assert(genomeData.getGenome());

        if (genomeData.isProtected())
        {
            continue;
        }

        for (ModifierPtr& modifier : m_modifiers)
        {
            modifier->modifyGenomes(genomeData.m_genome);
        }
    }

    // Evaluate all genomes.
    calcFitness();

    postUpdateGeneration();

    // Update the generation id.
    m_id = GenerationId(m_id.val() + 1);
}

void GenerationBase::calcFitness()
{
    assert(m_fitnessCalculator);

    for (GenomeData& gd : *m_genomes)
    {
        gd.setFitness(m_fitnessCalculator->calcFitness(*gd.getGenome()));
    }
}

void GenerationBase::addGenome(GenomeBasePtr genome, bool protectGenome)
{
    (*m_genomes)[m_numGenomes].init(genome, protectGenome, GenomeId(m_numGenomes));
    m_numGenomes++;
}
