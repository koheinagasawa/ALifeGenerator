/*
* CppnImageGen main.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <NEAT/NeuralNetwork/Activations/ActivationFactory.h>
#include <NEAT/NeuralNetwork/Activations/ActivationLibrary.h>
#include <NEAT/GeneticAlgorithms/Base/GenerationBase.h>
#include <NEAT/GeneticAlgorithms/NEAT/Generation.h>
#include <NEAT/GeneticAlgorithms/NEAT/Genome.h>
#include <bitmap_image.hpp>
#include <sstream>

using Image = std::vector<rgb_t>;

class ImageMatchingFitnessCalculator : public FitnessCalculatorBase
{
public:
    ImageMatchingFitnessCalculator(const Image& refImage, int xDim, int yDim)
        : m_xDim(xDim)
        , m_yDim(yDim)
        , m_referenceImage(refImage)
    {
        m_generatedImage.resize(xDim * yDim);
    }

    inline int coords2index(unsigned int x, unsigned int y)
    {
        return x * m_xDim + y;
    }

    void outputImage(const std::string& filename)
    {
        const unsigned int dim = 1000;

        bitmap_image image(m_xDim, m_yDim);

        for (unsigned int x = 0; x < m_xDim; ++x)
        {
            for (unsigned int y = 0; y < m_yDim; ++y)
            {
                const rgb_t& col = m_generatedImage[coords2index(x, y)];
                image.set_pixel(x, y, col.red, col.green, col.blue);
            }
        }

        std::string str;
        str.append(filename);
        str.append(".bmp");
        image.save_image(str.c_str());
    }

    void generateImage(const GenomeBase* genome)
    {
        std::vector<float> inputValues;
        inputValues.resize(2);
        for (unsigned int x = 0; x < m_xDim; x++)
        {
            for (unsigned int y = 0; y < m_yDim; y++)
            {
                inputValues[0] = (float)x;
                inputValues[1] = (float)y;
                evaluateGenome(genome, inputValues, 1.0f);
                const GenomeBase::Network* network = genome->getNetwork();
                float val = network->getNode(network->getOutputNodes()[0]).getValue();
                unsigned char charVal;
                if (val < 0)
                {
                    charVal = 0;
                }
                else if (val > 1.0f)
                {
                    charVal = 255;
                }
                else
                {
                    charVal = unsigned char(val * 255.f);
                }
                m_generatedImage[coords2index(x, y)].red = charVal;
                m_generatedImage[coords2index(x, y)].green = charVal;
                m_generatedImage[coords2index(x, y)].blue = charVal;
            }
        }
    }

    bool test(const GenomeBase* genome)
    {
        float fitness = calcFitness(genome);
        return fitness > 200.f;
    }

    virtual float calcFitness(const GenomeBase* genome) override
    {
        generateImage(genome);
        float diff = 0.f;
        for (unsigned int x = 0; x < m_xDim; x++)
        {
            for (unsigned int y = 0; y < m_yDim; y++)
            {
                int index = coords2index(x, y);
                diff += fabsf((float)(m_referenceImage[index].red - m_generatedImage[index].red));
            }
        }
        diff /= (float)(m_xDim * m_yDim);
        return 255.f - diff;
    }

    virtual FitnessCalcPtr clone() const override
    {
        return std::make_shared<ImageMatchingFitnessCalculator>(m_referenceImage, m_xDim, m_yDim);
    }

    static std::vector<rgb_t> getImage(const std::string& filename, int xDim, int yDim)
    {
        bitmap_image image(filename);

        if (!image)
        {
            printf("Error - Failed to open '%s'\n", filename.c_str());
            return;
        }

        Image imageOut;
        imageOut.resize(xDim * yDim);

        for (unsigned int x = 0; x < xDim; ++x)
        {
            for (unsigned int y = 0; y < yDim; ++y)
            {
                rgb_t& color = imageOut[x * xDim + y];
                image.get_pixel(x, y, color);
            }
        }

        return imageOut;
    }

protected:
    const unsigned int m_xDim;
    const unsigned int m_yDim;
    const Image& m_referenceImage;
    Image m_generatedImage;
};

int main()
{
    ActivationLibrary activationLib;
    {
        std::vector<ActivationFacotry::Type> activationTypes;
        activationTypes.push_back(ActivationFacotry::SIGMOID);
        activationTypes.push_back(ActivationFacotry::BIPOLAR_SIGMOID);
        activationTypes.push_back(ActivationFacotry::RELU);
        activationTypes.push_back(ActivationFacotry::GAUSSIAN);
        activationTypes.push_back(ActivationFacotry::LINEAR);
        activationTypes.push_back(ActivationFacotry::ABSOLUTE);
        activationTypes.push_back(ActivationFacotry::SINE);
        activationTypes.push_back(ActivationFacotry::COSINE);
        activationTypes.push_back(ActivationFacotry::TANGENT);
        activationTypes.push_back(ActivationFacotry::HYPERBOLIC_TANGENT);
        activationTypes.push_back(ActivationFacotry::RAMP);
        activationTypes.push_back(ActivationFacotry::STEP);
        activationTypes.push_back(ActivationFacotry::SPIKE);
        activationTypes.push_back(ActivationFacotry::INVERSE);
        activationLib.registerActivations(activationTypes);
    }

    RandomActivationProvider activationProvider(activationLib);

    Image refImage = ImageMatchingFitnessCalculator::getImage("Resource/CppnRefImage.bmp", 300, 300);
    auto fitnessCalculator = std::make_shared<ImageMatchingFitnessCalculator>(refImage, 300, 300);

    NEAT::Generation::Cinfo genCinfo;
    genCinfo.m_numGenomes = 150;
    genCinfo.m_genomeCinfo.m_numInputNodes = 2; // XY coordinates
    genCinfo.m_genomeCinfo.m_numOutputNodes = 1; // Gray scale value of the pixel
    genCinfo.m_genomeCinfo.m_createBiasNode = true;
    genCinfo.m_genomeCinfo.m_activationProvider = &activationProvider;
    genCinfo.m_mutationParams.m_activationProvider = &activationProvider;
    genCinfo.m_fitnessCalculator = fitnessCalculator;

    const int maxGeneration = 1000;

    std::cout << "Starting evolution ..." << std::endl;
    NEAT::InnovationCounter innovCounter;
    genCinfo.m_genomeCinfo.m_innovIdCounter = &innovCounter;
    NEAT::Generation generation(genCinfo);

    int i = 0;
    for (; i < maxGeneration; ++i)
    {
        std::cout << "Generation " << i << " ..." << std::endl;
        generation.evolveGeneration();
        const int numGeneration = generation.getId().val();

        std::shared_ptr<const GenomeBase> bestGenome = generation.getGenomesInFitnessOrder()[0].getGenome()->clone();
        const NEAT::Genome::Network* network = bestGenome->getNetwork();
        std::cout << "Number of total nodes: " << network->getNumNodes() << std::endl;
        std::cout << "Number of enabled edges: " << bestGenome->getNumEnabledEdges() << std::endl;
        std::cout << "=============================" << std::endl;

        if (fitnessCalculator->test(bestGenome.get()))
        {
            std::cout << "Solution Found at Generation " << numGeneration << "!" << std::endl;

            fitnessCalculator->outputImage("Output/result");
            break;
        }

        if (i % 10 == 0)
        {
            fitnessCalculator->generateImage(bestGenome.get());
            std::ostringstream oss;
            oss << "Output/gen" << i;
            fitnessCalculator->outputImage(oss.str());
        }
    }

    if (i == maxGeneration)
    {
        std::cout << "Failed! Reached to the max generation " << maxGeneration << std::endl;
    }

    return 0;
}
