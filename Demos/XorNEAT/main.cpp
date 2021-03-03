/*
* XorNEAT main.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/GeneticAlgorithms/NEAT/Generation.h>

int main()
{
    using namespace NEAT;

    Genome::Activation sigmoid = [](float value)
    {
        return 1.f / (1.f + exp(-4.9f * value));
    };
    sigmoid.m_name = "sigmoid";

    InnovationCounter innovCounter;

    class XorFitnessCalculator : public FitnessCalculatorBase
    {
    public:
        virtual float calcFitness(const GenomeBase& genome) const override
        {
            float score = 0.f;

            // Test 4 patterns of XOR
            score += EvaluateImpl(genome, false, false);
            score += 1.0f - EvaluateImpl(genome, false, true);
            score += 1.0f - EvaluateImpl(genome, true, false);
            score += EvaluateImpl(genome, true, true);
            score = 4.0f - score;

            return score * score;
        }
    };

    Generation::Cinfo genCinfo;
    genCinfo.m_numGenomes = 100;
    genCinfo.m_genomeCinfo.m_numInputNodes = 3; // Two inputs for XOR and one bias node.
    genCinfo.m_genomeCinfo.m_numOutputNodes = 1;
    genCinfo.m_genomeCinfo.m_defaultActivation = &sigmoid;
    genCinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;

    Generation generation(genCinfo);

    return 0;
}
