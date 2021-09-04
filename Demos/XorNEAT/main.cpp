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

    virtual float calcFitness(GenomeBase* genome) override
    {
        m_numEvaluations++;
        float score = 0.f;

        // Test 4 patterns of XOR
        score += fabs(evaluate(genome, false, false));
        score += fabs(1.0f - evaluate(genome, false, true));
        score += fabs(1.0f - evaluate(genome, true, false));
        score += fabs(evaluate(genome, true, true));
        score = 4.0f - score;

        return score * score;
    }

    virtual FitnessCalcPtr clone() const override
    {
        return std::make_shared<XorFitnessCalculator>();
    }

    float evaluate(GenomeBase* genome, bool input1, bool input2)
    {
        // Initialize values
        std::vector<float> values;
        values.resize(2);
        values[0] = input1 ? 1.f : 0.f;
        values[1] = input2 ? 1.f : 0.f;

        evaluateGenome(genome, values, 1.0f);

        return genome->getNodeValue(genome->getOutputNodes()[0]);
    }

    bool test(GenomeBase* genome)
    {
        bool result = true;
        // Test 4 patterns of XOR
        result &= evaluate(genome, false, false) < 0.5f;
        result &= evaluate(genome, false, true) >= 0.5f;
        result &= evaluate(genome, true, false) >= 0.5f;
        result &= evaluate(genome, true, true) < 0.5f;

        return result;
    }

    int m_numEvaluations = 0;
};

int main()
{
    using namespace NEAT;

    DefaultActivationProvider sigmoid([](float value) { return 1.f / (1.f + expf(-4.9f * value)); }, "sigmoid");


    auto fitnessCalc = std::make_shared<XorFitnessCalculator>();

    Generation::Cinfo genCinfo;
    genCinfo.m_numGenomes = 150;
    genCinfo.m_genomeCinfo.m_numInputNodes = 2; // Two inputs for XOR.
    genCinfo.m_genomeCinfo.m_numOutputNodes = 1;
    genCinfo.m_genomeCinfo.m_createBiasNode = true;
    genCinfo.m_genomeCinfo.m_activationProvider = &sigmoid;
    genCinfo.m_fitnessCalculator = fitnessCalc;
    genCinfo.m_mutationParams.m_activationProvider = &sigmoid;
    genCinfo.m_numThreads = 1;

    // Variables for performance investigation
    const int maxGeneration = 200;
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

        InnovationCounter innovCounter;
        genCinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;

        Generation generation(genCinfo);

        int i = 0;
        for (; i < maxGeneration; ++i)
        {
            generation.evolveGeneration();
            const int numGeneration = generation.getId().val();

            std::shared_ptr<GenomeBase> bestGenome = generation.getGenomesInFitnessOrder()[0].getGenome()->clone();
            if (fitnessCalc->test(bestGenome.get()))
            {
                std::cout << "Solution Found at Generation " << numGeneration << "!" << std::endl;

                // Get data for performance investigation
                totalGenerations += numGeneration;
                if (worstGenerations < numGeneration)
                {
                    worstGenerations = numGeneration;
                }
                totalNumHiddenNodes += bestGenome->getNumNodes() - 4; // 4 is two inputs, one output and one bias
                totalNumNondisabledConnections += bestGenome->getNumEnabledEdges();

                int numEvaluation = 0;
                for (const auto& calc : generation.getFitnessCalculators())
                {
                    numEvaluation += static_cast<const XorFitnessCalculator*>(calc.get())->m_numEvaluations;
                }

                if (worstEvaluationCount < numEvaluation)
                {
                    worstEvaluationCount = numEvaluation;
                }
                totalEvaluationCount += numEvaluation;

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
