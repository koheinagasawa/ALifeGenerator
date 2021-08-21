/*
* GenerationBase.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/GeneticAlgorithms/Base/GenerationBase.h>
#include <NEAT/GeneticAlgorithms/Base/Generators/GenomeGenerator.h>
#include <NEAT/GeneticAlgorithms/Base/Modifiers/GenomeModifier.h>

#include <omp.h>

//
// FitnessCalculatorBase
//

void FitnessCalculatorBase::evaluateGenome(const GenomeBase* genome, const std::vector<float>& inputNodeValues, float biasNodeValue)
{
    genome->clearNodeValues();
    genome->setInputNodeValues(inputNodeValues, biasNodeValue);
    m_evaluator.evaluate(genome->accessNetwork().get());
}

//
// GenerationBase::GenomeData
//

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

//
// GenerationBase
//

GenerationBase::GenerationBase(GenerationId id, int numGenomes, RandomGenerator* randomGenerator)
    : m_randomGenerator(randomGenerator)
    , m_numGenomes(numGenomes)
    , m_id(id)
{
    assert(m_numGenomes > 0);
}

void GenerationBase::createFitnessCalculators(FitnessCalcPtr fitnessCalc, int numThreads)
{
    m_fitnessCalculators.resize(numThreads);
    for (FitnessCalcPtr& calc : m_fitnessCalculators)
    {
        calc = fitnessCalc->clone();
    }
}

void GenerationBase::evolveGeneration()
{
    // TODO: Profile each process by adding timers.
    assert(m_genomes);
    assert(m_generators.size() > 0);
    const int numGenomes = getNumGenomes();
    assert(numGenomes > 1);
    assert(m_fitnessCalculators.size() > 0 && m_fitnessCalculators[0]);

    preUpdateGeneration();

    // Create a genome selector
    GenomeSelectorPtr selector = createSelector();

    // Swap the current generation and the previous generation.
    std::swap(m_genomes, m_prevGenGenomes);

    // Allocate buffer of GenomeData if it's not there yet.
    {
        if (!m_genomes)
        {
            m_genomes = std::make_shared<GenomeDatas>();
        }
        if (m_genomes->size() != numGenomes)
        {
            m_genomes->resize(numGenomes);
        }
    }

    int numGenomesToAdd = numGenomes;
    m_numGenomes = 0;

    // Create genomes for new generations by applying each genome generators.
    for (GeneratorPtr& generator : m_generators)
    {
        // [todo] Add a way to notify generator that if it's the last generator in this generation
        //        so that it can generate all the remaining genomes.
        generator->generate(numGenomes, numGenomesToAdd, selector.get());

        // Add generated genomes to the population.
        const bool protectGenomes = generator->shouldGenomesProtected();
        for (auto& newGenome : generator->getGeneratedGenomes())
        {
            (*m_genomes)[m_numGenomes].init(newGenome, protectGenomes, GenomeId(m_numGenomes));
            m_numGenomes++;
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
            // Skip protected genome.
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

namespace
{
    inline float calcFitnessImpl(const GenomeBase* genome, FitnessCalculatorBase* calculator, float& bestFitness)
    {
        float fitness = calculator->calcFitness(genome);
        if (fitness > bestFitness)
        {
            bestFitness = fitness;
        }
        return fitness;
    }
}

void GenerationBase::calcFitness()
{
    assert(m_fitnessCalculators.size() > 0 && m_fitnessCalculators[0]);

    m_bestFitness = 0;

    const int numThreads = (int)m_fitnessCalculators.size();
    if (numThreads > 1)
    {
        // Multi-threaded evaluation

        // Distribute evaluation tasks to each thread.
        int genomesPerThread = (int)(m_genomes->size() / numThreads);

        #pragma omp parallel for
        for (int threadId = 0; threadId < numThreads; threadId++)
        {
            FitnessCalcPtr calculator = m_fitnessCalculators[threadId];
            const int offset = threadId * genomesPerThread;
            for (int i = 0; i < genomesPerThread; i++)
            {
                GenomeData& gd = (*m_genomes)[i + offset];
                gd.setFitness(calcFitnessImpl(gd.m_genome.get(), calculator.get(), m_bestFitness));
            }
        }

        // Run remaining evaluation tasks.
        const int offset = numThreads * genomesPerThread;
        #pragma omp parallel for
        for (int i = offset; i < (int)m_genomes->size(); i++)
        {
            const int threadId = i - offset;
            assert(threadId < (int)m_fitnessCalculators.size());
            GenomeData& gd = (*m_genomes)[i];
            gd.setFitness(calcFitnessImpl(gd.m_genome.get(), m_fitnessCalculators[threadId].get(), m_bestFitness));
        }
    }
    else
    {
        // Single-threaded evaluation
        FitnessCalcPtr calculator = m_fitnessCalculators[0];
        for (GenomeData& gd : *m_genomes)
        {
            gd.setFitness(calcFitnessImpl(gd.m_genome.get(), calculator.get(), m_bestFitness));
        }
    }
}
