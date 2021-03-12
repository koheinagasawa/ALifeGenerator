/*
* SpeciesBasedGenomeSelectorTest.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <UnitTest/UnitTestPch.h>

#include <NEAT/GeneticAlgorithms/NEAT/Selectors/SpeciesBasedGenomeSelector.h>

namespace
{
    // Custom random generator.
    class MyRandom : public PseudoRandom
    {
    public:
        MyRandom() : PseudoRandom(0) {}

        virtual float randomReal(float min, float max) override 
        {
            float v = min + m_val;
            m_val += 1.f;
            return std::min(max, v);
        }

        virtual float randomReal01() override
        {
            return 0.f;
        }

        void reset() { m_val = 0; }

        float m_val;
    };
}

TEST(SpeciesBasedGenomeSelector, CreateSelector)
{
    using namespace NEAT;
    using GenomePtr = std::shared_ptr<Genome>;
    using GenomeData = GenerationBase::GenomeData;
    using GenomeDatas = GenerationBase::GenomeDatas;
    using SpeciesPtr = SpeciesBasedGenomeSelector::SpeciesPtr;
    using SpeciesList = SpeciesBasedGenomeSelector::SpeciesList;
    using GenomeSpeciesMap = SpeciesBasedGenomeSelector::GenomeSpeciesMap;

    // Create data storage we need to create DefaultGenomeSelector.
    GenomeDatas genomes;
    SpeciesList species;
    GenomeSpeciesMap genomeSpeciesMap;

    // Create a genome.
    InnovationCounter innovCounter;
    Genome::Cinfo cinfo;
    {
        cinfo.m_numInputNodes = 2;
        cinfo.m_numOutputNodes = 2;
        cinfo.m_innovIdCounter = &innovCounter;
    }
    GenomePtr genome0 = std::make_shared<Genome>(cinfo);
    genomes.push_back({genome0, GenomeId(0) });
    genomes.back().setFitness(1.f);

    // Create a selector with no species.
    // You can select one genome selectTwoGenomes shouldn't work.
    {
        SpeciesBasedGenomeSelector selector(genomes, species, genomeSpeciesMap);
        EXPECT_EQ(selector.getNumGenomes(), 1);
        EXPECT_EQ(selector.selectGenome(), &genomes[0]);
        const GenomeData* g1 = nullptr;
        const GenomeData* g2 = nullptr;
        selector.selectTwoGenomes(g1, g2);
        EXPECT_EQ(g1, nullptr);
        EXPECT_EQ(g2, nullptr);
    }

    // Create more genomes.
    for (int i = 1; i < 5; i++)
    {
        GenomePtr g = std::make_shared<Genome>(*genome0);
        genomes.push_back({ g, GenomeId(i) });
        genomes.back().setFitness((float)(i + 1));
    }

    // Create species.
    {
        Genome::CalcDistParams calcDistParams;
        {
            calcDistParams.m_disjointFactor = 1.0f;
            calcDistParams.m_weightFactor = 1.0f;
        }
        SpeciesPtr s1 = std::make_shared<Species>(std::static_pointer_cast<const Genome>(genomes[0].getGenome()), genomes[0].getFitness());
        s1->tryAddGenome(std::static_pointer_cast<const Genome>(genomes[1].getGenome()), genomes[1].getFitness(), 1000.0f, calcDistParams);

        SpeciesPtr s2 = std::make_shared<Species>(std::static_pointer_cast<const Genome>(genomes[2].getGenome()), genomes[2].getFitness());
        s2->tryAddGenome(std::static_pointer_cast<const Genome>(genomes[3].getGenome()), genomes[3].getFitness(), 1000.0f, calcDistParams);
        s2->tryAddGenome(std::static_pointer_cast<const Genome>(genomes[4].getGenome()), genomes[4].getFitness(), 1000.0f, calcDistParams);

        species.insert({ SpeciesId(0), s1 });
        species.insert({ SpeciesId(1), s2 });
    }

    // Create a selector with empty map.
    // Species will affect adjusted fitness of each genome but shouldn't affect selection itself.
    {
        MyRandom random;
        random.reset();
        SpeciesBasedGenomeSelector selector(genomes, species, genomeSpeciesMap, &random);
        EXPECT_EQ(selector.getNumGenomes(), 5);
        EXPECT_EQ(selector.selectGenome(), &genomes[0]);

        random.reset();

        const GenomeData* g1 = nullptr;
        const GenomeData* g2 = nullptr;

        selector.setInterSpeciesSelectionRate(1.0f);
        selector.selectTwoGenomes(g1, g2);
        EXPECT_EQ(g1, &genomes[0]);
        EXPECT_EQ(g2, &genomes[1]);

        random.reset();

        selector.setInterSpeciesSelectionRate(0.0f);
        selector.selectTwoGenomes(g1, g2);
        EXPECT_EQ(g1, &genomes[0]);
        EXPECT_EQ(g2, &genomes[1]);
    }

    // Create a map
    genomeSpeciesMap.insert({ GenomeId(0), SpeciesId(0) });
    genomeSpeciesMap.insert({ GenomeId(1), SpeciesId(0) });
    genomeSpeciesMap.insert({ GenomeId(2), SpeciesId(1) });
    genomeSpeciesMap.insert({ GenomeId(3), SpeciesId(1) });
    genomeSpeciesMap.insert({ GenomeId(4), SpeciesId(1) });

    // Create a selector.
    // Selection should be affected by Species.
    {
        MyRandom random;
        random.reset();
        SpeciesBasedGenomeSelector selector(genomes, species, genomeSpeciesMap, &random);
        EXPECT_EQ(selector.getNumGenomes(), 5);
        EXPECT_EQ(selector.selectGenome(), &genomes[0]);

        random.reset();

        const GenomeData* g1 = nullptr;
        const GenomeData* g2 = nullptr;

        // Allow inter-species selection
        selector.setInterSpeciesSelectionRate(1.0f);
        selector.selectTwoGenomes(g1, g2);
        EXPECT_EQ(g1, &genomes[0]);
        EXPECT_EQ(g2, &genomes[1]);

        random.m_val = 1.0f;

        selector.selectTwoGenomes(g1, g2);
        EXPECT_EQ(g1, &genomes[1]);
        EXPECT_EQ(g2, &genomes[2]);

        random.m_val = 1.0f;

        // Disallow inter-species selection
        selector.setInterSpeciesSelectionRate(0.0f);
        selector.selectTwoGenomes(g1, g2);
        EXPECT_EQ(g1, &genomes[0]);
        EXPECT_EQ(g2, &genomes[1]);

        random.m_val = 3.0f;

        selector.selectTwoGenomes(g1, g2);
        EXPECT_EQ(g1, &genomes[3]);
        EXPECT_EQ(g2, &genomes[4]);
    }
}
