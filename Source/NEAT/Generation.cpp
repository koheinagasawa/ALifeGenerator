/*
* Generation.h
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/Neat.h>
#include <NEAT/Generation.h>

#include <limits>

using namespace NEAT;

namespace
{
    // Helper class to select a random genome by taking fitness into account.
    class GenomeSelector
    {
    public:

        // Constructor
        GenomeSelector(PseudoRandom& random)
            : m_random(m_random)
        {
        }

        // Set genomes to select and initialize internal data.
        bool setGenomes(const Generation::GenomeDatas& genomesIn, const Generation::SpeciesList& species)
        {
            assert(genomesIn.size() > 0);

            m_genomes.reserve(genomesIn.size());
            m_sumFitness.reserve(genomesIn.size()+1);
            float sumFitness = 0;
            m_sumFitness.push_back(0);

            int currentSpecies = genomesIn[0].getSpeciesIndex();
            float invCurrentSpeciesNumMembers = 1.f / (float)species[currentSpecies].getNumMembers();
            int currentSpeciesStartIndex = 0;

#ifdef _DEBUG
            std::unordered_set<int> speciesIndices;
            speciesIndices.insert(currentSpecies);
#endif

            for (const Generation::GenomeData& g : genomesIn)
            {
                if (g.canReproduce())
                {
                    m_genomes.push_back(&g);

                    if (currentSpecies != g.getSpeciesIndex())
                    {
                        m_spciecesStartEndIndices[currentSpecies] = { currentSpeciesStartIndex, (int)m_genomes.size() };

                        currentSpecies = g.getSpeciesIndex();
                        invCurrentSpeciesNumMembers = 1.f / (float)species[currentSpecies].getNumMembers();
                        currentSpeciesStartIndex = (int)m_genomes.size();
#ifdef _DEBUG
                        assert(speciesIndices.find(currentSpecies) == speciesIndices.end());
#endif
                    }

                    const float adjustedFitness = g.getFitness() * invCurrentSpeciesNumMembers;
                    sumFitness += adjustedFitness;
                    m_sumFitness.push_back(sumFitness);
                }
            }

            if (m_genomes.size() == 0 || sumFitness == 0)
            {
                return false;
            }

            m_spciecesStartEndIndices[currentSpecies] = { currentSpeciesStartIndex, (int)m_genomes.size() };

            assert(m_genomes.size() + 1 == m_sumFitness.size());

            return true;
        }

        // Select a random genome.
        auto selectRandomGenome()->const Generation::GenomeData*
        {
            return selectRandomGenome(0, m_genomes.size());
        }

        // Select two random genomes in the same species.
        void selectTwoRandomGenomesInSameSpecies(const Generation::GenomeData*& g1, const Generation::GenomeData*& g2)
        {
            assert(m_spciecesStartEndIndices.size() > 0);

            g1 = nullptr;
            g2 = nullptr;

            if (m_genomes.size() <= 1 || (m_genomes.size() - m_spciecesStartEndIndices.size()) <= 1)
            {
                // None of species has more than one genomes.
                return;
            }

            while (1)
            {
                // Select a random genome.
                g1 = selectRandomGenome(0, m_genomes.size());
                assert(g1);

                // Get start and end indices of the species of g1.
                const IndexSet& startEnd = m_spciecesStartEndIndices.at(g1->getSpeciesIndex());

                if (startEnd.m_end - startEnd.m_start < 2)
                {
                    // There are only one genome in this species. Try to select different species.
                    continue;
                }

                // Select g2 among the species.
                g2 = g1;
                while (g1 == g2)
                {
                    g2 = selectRandomGenome(startEnd.m_start, startEnd.m_end);
                }

                break;
            }

            assert(g1->getSpeciesIndex() == g2->getSpeciesIndex());
        }

    private:
        // Select a random genome between start and end (not including end) in m_genomes array.
        const Generation::GenomeData* selectRandomGenome(int start, int end)
        {
            assert(m_genomes.size() > 0 && (m_genomes.size() + 1 == m_sumFitness.size()));
            assert(start >= 0 &&  end < (int)m_sumFitness.size() && start < end);

            const float v = m_random.randomReal(m_sumFitness[start], m_sumFitness[end] - std::numeric_limits<float>::epsilon());
            for (int i = start; i < end; i++)
            {
                if (v < m_sumFitness[i+1])
                {
                    return m_genomes[i];
                }
            }

            assert(0);
            return nullptr;
        }

        struct IndexSet
        {
            int m_start;
            int m_end;
        };

        std::vector<const Generation::GenomeData*> m_genomes;
        std::vector<float> m_sumFitness;
        std::unordered_map<int, IndexSet> m_spciecesStartEndIndices;
        PseudoRandom& m_random;
    };
}

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

    const int numGenomes = getNumGenomes();
    assert(numGenomes > 1);

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

    // Select genomes which are copied to the next generation unchanged
    for (const Species& species : m_species)
    {
        if (species.getStagnantGenerationCount() >= params.m_maxStagnantCount)
        {
            continue;
        }

        if (species.getNumMembers() >= params.m_minMembersInSpeciesToCopyChampion)
        {
            addGenomeToNewGen(species.getBestGenome());
        }
    }

    GenomeSelector selector(random);
    selector.setGenomes(*m_prevGenGenomes, m_species);

    // Select and mutate genomes
    {
        const int numGenomesToSelect = std::min(numGenomesToAdd, int(numGenomes * (1.f - params.m_crossOverRate)));
        Genome::MutationOut mout;
        int i = 0;
        while (i < numGenomesToSelect)
        {
            // Select a random genome.
            const GenomeData* gd = selector.selectRandomGenome();

            assert(gd->canReproduce());

            // Copy genome in this generation first.
            GenomePtr copy = std::make_shared<Genome>(*gd->m_genome);

            // Mutate the genome.
            copy->mutate(params.m_mutationParams, mout);

            addGenomeToNewGen(copy);
            i++;
        }
    }

    // Select and generate new genomes by crossover
    {
        int numGenomesToInterSpeciesCO = (int)(numGenomesToAdd * params.m_interSpeciesCrossOverRate);
        int numGenomesToIntraSpeciesCO = (int)(numGenomesToAdd - numGenomesToInterSpeciesCO);

        auto crossOver = [&addGenomeToNewGen, &params](const GenomeData* g1, const GenomeData* g2)
        {
            assert(g1 && g2 && g1->canReproduce() && g2->canReproduce());
            bool isSameFitness = g1->getFitness() == g2->getFitness();
            GenomePtr newGenome = std::make_shared<Genome>(Genome::crossOver(*g1->m_genome, *g2->m_genome, isSameFitness, params.m_crossOverParams));
            addGenomeToNewGen(newGenome);
        };

        for (int i = 0; i < numGenomesToIntraSpeciesCO; i++)
        {
            const GenomeData* g1 = nullptr;
            const GenomeData* g2 = nullptr;

            selector.selectTwoRandomGenomesInSameSpecies(g1, g2);

            if (g1 == nullptr)
            {
                // Failed to select two genomes in the same species, which means
                // we have no species which has more than one genome in it.
                // Just fall back to doing inter species cross over for all the genomes.
                assert(i == 0);
                numGenomesToInterSpeciesCO = numGenomesToAdd;
                break;
            }

            crossOver(g1, g2);
        }

        for (int i = 0; i < numGenomesToInterSpeciesCO; i++)
        {
            const GenomeData* g1 = selector.selectRandomGenome();
            const GenomeData* g2 = g1;
            while (g1 == g2)
            {
                g2 = selector.selectRandomGenome();
            }

            crossOver(g1, g2);
        }
    }

    // We should have added all the genomes at this point.
    assert(m_genomes->size() == m_prevGenGenomes->size());

    // Evaluate all genomes
    for (GenomeData& gd : *m_genomes)
    {
        gd.m_fitness = m_fitnessCalculator->calcFitness(*gd.m_genome);
    }

    // Speciation
    {
        // Remove stagnant species and empty species first
        {
            auto itr = m_species.begin();
            while (itr != m_species.end())
            {
                if (itr->getNumMembers() == 0 ||  itr->getStagnantGenerationCount() >= params.m_maxStagnantCount)
                {
                    itr = m_species.erase(itr);
                }
                else
                {
                    itr++;
                }
            }
        }

        for (Species& s : m_species)
        {
            s.preNewGeneration(&random);
        }

        for (GenomeData& gd : *m_genomes)
        {
            // Try to find a species
            int i = 0;
            for (; i < (int)m_species.size(); i++)
            {
                if (m_species[i].tryAddGenome(gd.m_genome, gd.m_fitness, params.m_speciationDistanceThreshold, params.m_calcDistParams))
                {
                    gd.m_speciesIndex = i;
                    break;
                }
            }

            if (i == (int)m_species.size())
            {
                // No species found. Create a new one for this genome.
                gd.m_speciesIndex = i;
                m_species.push_back(Species(*gd.m_genome));
                m_species.back().tryAddGenome(gd.m_genome, gd.m_fitness, params.m_speciationDistanceThreshold, params.m_calcDistParams);
            }
        }

        for (Species& s : m_species)
        {
            s.postNewGeneration();
        }
    }

    // Mark genomes which shouldn't reproduce anymore
    for(GenomeData& gd : *m_genomes)
    {
        gd.m_canReproduce = m_species[gd.getSpeciesIndex()].getStagnantGenerationCount() < params.m_maxStagnantCount;
    }

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