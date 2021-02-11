/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/Generation.h>

using namespace NEAT;

Generation::GenomeData::GenomeData(GenomePtr genome, GenomeId id)
    : m_genome(genome)
    , m_id(id)
{
}

void Generation::GenomeData::init(GenomePtr genome, GenomeId id)
{
    m_genome = genome;
    m_id = id;
    m_fitness = 0.f;
    m_canReproduce = true;
}

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
        GenomePtr genome = std::make_shared<Genome>(cinfo.m_genomeCinfo);

        // Randomize edge weights
        const Genome::Network* network = genome->getNetwork();
        for (auto itr : network->getEdges())
        {
            genome->setEdgeWeight(itr.first, random.randomReal(cinfo.m_minWeight, cinfo.m_maxWeight));
        }

        m_genomes->push_back(GenomeData(genome, GenomeId(i)));
    }
    m_numGenomes = m_genomes->size();

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
            m_genomes->push_back(GenomeData(genome, id));
            id = id.val() + 1;
        }
    }
    m_numGenomes = m_genomes->size();

    // Create one species
    {
        PseudoRandom& random = PseudoRandom::getInstance();
        const GenomeData& representative = (*m_genomes)[random.randomInteger(0, m_genomes->size() - 1)];
        m_species.push_back(Species(*representative.m_genome));
    }
}

void Generation::createNewGeneration(const CreateNewGenParams& params)
{
    PseudoRandom& random = params.m_random ? *params.m_random : PseudoRandom::getInstance();

    const int numGenomes = getNumGenomes();;
    int numGenomesToAdd = numGenomes;
    std::swap(m_genomes, m_prevGenGenomes);

    m_numGenomes = 0;

    // Helper function to add a new genome to the new generation
    auto addGenomeToNewGen = [this, &numGenomesToAdd](GenomePtr genome)
    {
        addGenome(genome);
        numGenomesToAdd--;
    };

    // Allocate buffer of GenomeData if it's not there yet.
    if (m_genomes->size() != numGenomes)
    {
        m_genomes->resize(numGenomes);
    }

    // Remove stagnant species first
    {
        auto itr = m_species.begin();
        while (itr != m_species.end())
        {
            if (itr->getStagnantGenerationCount() >= params.m_maxStagnantCount)
            {
                itr = m_species.erase(itr);
            }
            else
            {
                itr++;
            }
        }
    }

    // Select genomes which are copied to the next generation unchanged
    for (const Species& species : m_species)
    {
        assert(species.getStagnantGenerationCount() < params.m_maxStagnantCount);

        if (species.getNumMembers() >= params.m_minMembersInSpeciesToCopyChampion)
        {
            addGenomeToNewGen(species.getBestGenome());
        }
    }

    // Select and mutate genomes
    {
        const int numGenomesToSelect = std::min(numGenomesToAdd, int(numGenomes * (1.f - params.m_crossOverRate)));
        Genome::MutationOut mout;
        int i = 0;
        while (i < numGenomesToSelect)
        {
            // Select a random genome.
            // TODO Take fitness into account.
            const GenomeData& gd = (*m_prevGenGenomes)[random.randomInteger(0, numGenomes - 1)];

            if (!gd.canReproduce()) continue;

            // Copy genome in this generation first.
            GenomePtr copy = std::make_shared<Genome>(*gd.m_genome);

            // Mutate the genome.
            copy->mutate(params.m_mutationParams, mout);

            addGenomeToNewGen(copy);
            i++;
        }
    }

    // Select and generate new genomes by crossover
    {
        const int numGenomesToCrossOver = std::min(numGenomesToAdd, int(numGenomes * params.m_crossOverRate));
        for (int i = 0; i < numGenomesToCrossOver; i++)
        {

        }
    }

    // Speciation

    // Evaluate all genomes

    // Mark genomes which shouldn't reproduce anymore

    // Update the generation id.
    m_id = GenerationId(m_id.val() + 1);
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

void Generation::addGenome(GenomePtr genome)
{
    (*m_genomes)[m_numGenomes].init(genome, GenomeId(m_numGenomes));
    m_numGenomes++;
}
