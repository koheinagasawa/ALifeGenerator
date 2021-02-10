/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/Generation.h>

using namespace NEAT;

Generation::Generation(const Cinfo& cinfo)
    : m_id(GenerationId(0))
    , m_fitnessCalculator(cinfo.m_fitnessCalculator)
{
    assert(cinfo.m_numGenomes > 0);
    assert(cinfo.m_minWeight <= cinfo.m_maxWeight);
    assert(m_fitnessCalculator);

    PseudoRandom& random = cinfo.m_random ? *cinfo.m_random : PseudoRandom::getInstance();

    // Create genomes of the first generation.
    m_genomes = std::make_shared<GenomeDatas>();
    m_genomes->reserve(cinfo.m_numGenomes);
    for (int i = 0; i < cinfo.m_numGenomes; i++)
    {
        GenomeData gd;
        gd.m_genome = std::make_shared<Genome>(cinfo.m_genomeCinfo);
        gd.m_id = GenomeId(i);

        // Randomize edge weights
        const Genome::Network* network = gd.m_genome->getNetwork();
        for (auto itr : network->getEdges())
        {
            gd.m_genome->setEdgeWeight(itr.first, random.randomReal(cinfo.m_minWeight, cinfo.m_maxWeight));
        }

        m_genomes->push_back(gd);
    }

    // Create one species
    {
        const GenomeData& representative = (*m_genomes)[random.randomInteger(0, m_genomes->size() - 1)];
        m_species.push_back(Species(*representative.m_genome));
    }
}

Generation::Generation(const Genomes& genomes, FitnessCalculator* fitnessCalculator)
    : m_id(GenerationId(0))
    , m_fitnessCalculator(fitnessCalculator)
{
    assert(genomes.size() > 0);
    assert(m_fitnessCalculator);

    m_genomes = std::make_shared<GenomeDatas>();

    // Create GenomeData for each given genome
    {
        m_genomes->reserve(genomes.size());
        GenomeId id(0);
        for (const GenomePtr& genome : genomes)
        {
            GenomeData gd;
            gd.m_genome = genome;
            gd.m_id = id;
            m_genomes->push_back(gd);

            id = id.val() + 1;
        }
    }

    // Create one species
    {
        PseudoRandom& random = PseudoRandom::getInstance();
        const GenomeData& representative = (*m_genomes)[random.randomInteger(0, m_genomes->size() - 1)];
        m_species.push_back(Species(*representative.m_genome));
    }
}

Generation::Generation(GenerationId id, FitnessCalculator* fitnessCalculator)
    : m_id(id)
    , m_fitnessCalculator(fitnessCalculator)
{
}

auto Generation::createNewGeneration(const CreateNewGenParams& params) const->Generation
{
    Generation newGen(GenerationId(m_id.val() + 1), m_fitnessCalculator);

    // Select genomes which are copied to the next generation unchanged
    if (m_id > 0)
    {
        // STARTFROMHERE
    }

    // Mutate genomes

    // Select genomes

    // Generate new genomes by crossover

    // Evaluate all genomes

    // Reset species

    // Speciation

    // Mark genomes which shouldn't reproduce anymore

    return newGen;
}

void Generation::setInputNodeValues(const std::vector<float>& values)
{
    assert(getNumGenomes() > 0);

    for (int i = 0; i < getNumGenomes(); i++)
    {
        GenomePtr& genome = (*m_genomes)[i].m_genome;
        genome->setInputNodeValues(values);
    }
}
