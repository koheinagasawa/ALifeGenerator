/*
* XorNEAT main.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <iostream>
#include <sstream>
#include <fstream>

#include <NEAT/GeneticAlgorithms/NEAT/Generation.h>

class XorFitnessCalculator : public FitnessCalculatorBase
{
public:
    virtual float calcFitness(const GenomeBase& genome) override
    {
        float score = 0.f;

        // Test 4 patterns of XOR
        score += abs(evaluate(genome, false, false));
        score += abs(1.0f - evaluate(genome, false, true));
        score += abs(1.0f - evaluate(genome, true, false));
        score += abs(evaluate(genome, true, true));
        score = 4.0f - score;

        return score * score;
    }

    float evaluate(const GenomeBase& genome, bool input1, bool input2)
    {
        m_numEvaluations++;

        // Initialize values
        static std::vector<float> values;
        values.resize(3);
        values[0] = input1 ? 1.f : 0.f;
        values[1] = input2 ? 1.f : 0.f;
        values[2] = 1.f; // bias node

        genome.evaluate(values);

        const GenomeBase::Network* network = genome.getNetwork();
        return network->getNode(network->getOutputNodes()[0]).getValue();
    }

    bool test(const std::shared_ptr<const GenomeBase>& genome)
    {
        bool result = true;
        // Test 4 patterns of XOR
        result &= evaluate(*genome, false, false) <= 0.5f;
        result &= evaluate(*genome, false, true) > 0.5f;
        result &= evaluate(*genome, true, false) > 0.5f;
        result &= evaluate(*genome, true, true) <= 0.5f;

        return result;
    }

    int m_numEvaluations = 0;
};

int main()
{
    using namespace NEAT;

    Genome::Activation sigmoid = Genome::Activation([](float value) { return 1.f / (1.f + exp(-4.9f * value)); });
    sigmoid.m_name = "sigmoid";


    auto fitnessCalc = std::make_shared<XorFitnessCalculator>();

    Generation::Cinfo genCinfo;
    genCinfo.m_numGenomes = 150;
    genCinfo.m_genomeCinfo.m_numInputNodes = 3; // Two inputs for XOR and one bias node.
    genCinfo.m_genomeCinfo.m_numOutputNodes = 1;
    genCinfo.m_genomeCinfo.m_defaultActivation = &sigmoid;
    genCinfo.m_fitnessCalculator = fitnessCalc;

    // Variables for performance investigation
    const int maxGeneration = 100;
    const int numRun = 100;
    int numFailed = 0;
    int totalGenerations = 0;
    int worstGenerations = 0;
    int totalNumHiddenNodes = 0;
    int totalNumNondisabledConnections = 0;
    int totalEvaluationCount = 0;
    int worstEvaluationCount = 0;

    for (int run = 0; run < numRun; ++run)
    {
        std::cout << "Starting Run" << run << "..." << std::endl;

        fitnessCalc->m_numEvaluations = 0;
        InnovationCounter innovCounter;
        genCinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;

        Generation generation(genCinfo);

        int i = 0;
        for (; i < maxGeneration; ++i)
        {
            std::cout << "Generation " << i << std::endl;
            generation.evolveGeneration();
            const int numGeneration = generation.getId().val();

            const Generation::GenomeData bestGenome = generation.getGenomesInFitnessOrder()[0];
            if (fitnessCalc->test(bestGenome.getGenome()))
            {
                std::cout << "Solution Found at Generation " << numGeneration << "!" << std::endl;

                const Genome::Network* network = bestGenome.getGenome()->getNetwork();

                // Get data for performance investigation
                totalGenerations += numGeneration;
                if (worstGenerations < numGeneration)
                {
                    worstGenerations = numGeneration;
                }
                totalNumHiddenNodes += network->getNumNodes() - 4; // 4 is two inputs, one output and one bias
                totalNumNondisabledConnections += network->getNumEnabledEdges();
                if (worstEvaluationCount < fitnessCalc->m_numEvaluations)
                {
                    worstEvaluationCount = fitnessCalc->m_numEvaluations;
                }
                totalEvaluationCount += fitnessCalc->m_numEvaluations;

                break;
            }
        }

        if (i == maxGeneration)
        {
            std::cout << "Failed!" << std::endl;
            ++numFailed;
        }
    }

    const float invNumSuccess = 1.0f / float(numRun - numFailed);

    // Output result
    std::stringstream ss;
    ss << "=============================" << std::endl;
    ss << "Average successful generation : " << totalGenerations * invNumSuccess << std::endl;
    ss << "Worst successful generation : " << worstGenerations << std::endl;
    ss << "Number of failed run : " << numFailed << std::endl;
    ss << "Average number of hidden nodes of solution genome : " << totalNumHiddenNodes * invNumSuccess << std::endl;
    ss << "Average number of non-disabled connections of solution genome : " << totalNumNondisabledConnections * invNumSuccess << std::endl;
    ss << "Average evaluation count : " << totalEvaluationCount * invNumSuccess << std::endl;
    ss << "Worst evaluation count : " << worstEvaluationCount << std::endl;
    ss << "=============================" << std::endl;
    std::cout << ss.str();
    std::ofstream ofs("result.txt");
    ofs << ss.str();
    ofs.close();

    return 0;
}
