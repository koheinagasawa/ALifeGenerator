/*
* CppnImageGen main.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <EvoAlgo/NeuralNetwork/Activations/ActivationFactory.h>
#include <EvoAlgo/NeuralNetwork/Activations/ActivationLibrary.h>
#include <EvoAlgo/GeneticAlgorithms/Base/GenerationBase.h>
#include <EvoAlgo/GeneticAlgorithms/NEAT/Generation.h>
#include <EvoAlgo/GeneticAlgorithms/NEAT/Genome.h>
#include <bitmap_image.hpp>
#include <sstream>

using Image = std::vector<rgb_t>;

static bool s_grayScale = false;

class ImageMatchingFitnessCalculator : public FitnessCalculatorBase
{
public:
    ImageMatchingFitnessCalculator(const Image& refImage, int xDim, int yDim)
        : m_xDim(xDim)
        , m_yDim(yDim)
        , m_referenceImage(refImage)
    {
        m_generatedImage.resize(xDim * yDim);
        m_evaluator.m_evalIterations = 3;
    }

    inline int coords2index(unsigned int x, unsigned int y)
    {
        return x * m_xDim + y;
    }

    void outputImage(const std::string& filename)
    {
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

    unsigned char floatToUCharColor(float val)
    {
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

        return charVal;
    }

    void generateImage(GenomeBase* genome)
    {
        std::vector<float> inputValues;
        inputValues.resize(2);
        for (unsigned int x = 0; x < m_xDim; x++)
        {
            for (unsigned int y = 0; y < m_yDim; y++)
            {
                inputValues[0] = (float)(x) / (float)m_xDim;
                inputValues[1] = (float)(y) / (float)m_yDim;
                evaluateGenome(genome, inputValues, 1.0f);
                unsigned char r = floatToUCharColor(genome->getNodeValue(genome->getOutputNodes()[0]));

                if (s_grayScale)
                {
                    m_generatedImage[coords2index(x, y)].red = r;
                    m_generatedImage[coords2index(x, y)].green = r;
                    m_generatedImage[coords2index(x, y)].blue = r;
                }
                else
                {
                    unsigned char g = floatToUCharColor(genome->getNodeValue(genome->getOutputNodes()[1]));
                    unsigned char b = floatToUCharColor(genome->getNodeValue(genome->getOutputNodes()[2]));
                    m_generatedImage[coords2index(x, y)].red = r;
                    m_generatedImage[coords2index(x, y)].green = g;
                    m_generatedImage[coords2index(x, y)].blue = b;
                }


            }
        }
    }

    virtual float calcFitness(GenomeBase* genome) override
    {
        generateImage(genome);
        float diff = 0.f;
        for (unsigned int x = 0; x < m_xDim; x++)
        {
            for (unsigned int y = 0; y < m_yDim; y++)
            {
                int index = coords2index(x, y);
                if (s_grayScale)
                {
                    diff += fabsf((float)(m_referenceImage[index].red - m_generatedImage[index].red));
                }
                else
                {
                    diff += fabsf((float)(m_referenceImage[index].red - m_generatedImage[index].red));
                    diff += fabsf((float)(m_referenceImage[index].green - m_generatedImage[index].green));
                    diff += fabsf((float)(m_referenceImage[index].blue - m_generatedImage[index].blue));
                }
            }
        }

        if (s_grayScale)
        {
            diff /= (float)(m_xDim * m_yDim);
        }
        else
        {
            diff /= (float)(m_xDim * m_yDim * 3);
        }
        return 255.f - diff;
    }

    virtual FitnessCalcPtr clone() const override
    {
        return std::make_shared<ImageMatchingFitnessCalculator>(m_referenceImage, m_xDim, m_yDim);
    }

    static std::vector<rgb_t> getImage(const std::string& filename, int xDim, int yDim)
    {
        Image imageOut;
        bitmap_image image(filename);

        if (!image)
        {
            printf("Error - Failed to open '%s'\n", filename.c_str());
            return imageOut;
        }

        imageOut.resize(xDim * yDim);

        for (int x = 0; x < xDim; ++x)
        {
            for (int y = 0; y < yDim; ++y)
            {
                rgb_t& color = imageOut[x * xDim + y];
                image.get_pixel((unsigned)x, (unsigned)y, color);
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
        activationTypes.push_back(ActivationFacotry::AF_SIGMOID);
        activationTypes.push_back(ActivationFacotry::AF_BIPOLAR_SIGMOID);
        activationTypes.push_back(ActivationFacotry::AF_RELU);
        activationTypes.push_back(ActivationFacotry::AF_GAUSSIAN);
        activationTypes.push_back(ActivationFacotry::AF_ABSOLUTE);
        activationTypes.push_back(ActivationFacotry::AF_SINE);
        activationTypes.push_back(ActivationFacotry::AF_COSINE);
        activationTypes.push_back(ActivationFacotry::AF_HYPERBOLIC_TANGENT);
        activationTypes.push_back(ActivationFacotry::AF_RAMP);
        activationTypes.push_back(ActivationFacotry::AF_STEP);
        activationTypes.push_back(ActivationFacotry::AF_SPIKE);
        activationTypes.push_back(ActivationFacotry::AF_INVERSE);
        activationTypes.push_back(ActivationFacotry::AF_IDENTITY);
        activationTypes.push_back(ActivationFacotry::AF_CLAMPED);
        activationTypes.push_back(ActivationFacotry::AF_LOGARITHMIC);
        activationTypes.push_back(ActivationFacotry::AF_EXPONENTIAL);
        activationTypes.push_back(ActivationFacotry::AF_HAT);
        activationTypes.push_back(ActivationFacotry::AF_SQUARE);
        activationTypes.push_back(ActivationFacotry::AF_CUBE);
        activationLib.registerActivations(activationTypes);
    }

    RandomActivationProvider activationProvider(activationLib);

    const int pixelSize = 150;
    Image refImage = ImageMatchingFitnessCalculator::getImage("Resource/CppnRefImage.bmp", pixelSize, pixelSize);
    auto fitnessCalculator = std::make_shared<ImageMatchingFitnessCalculator>(refImage, pixelSize, pixelSize);

    NEAT::Generation::Cinfo genCinfo;
    genCinfo.m_numGenomes = 500;
    genCinfo.m_genomeCinfo.m_numInputNodes = 2; // XY coordinates
    genCinfo.m_genomeCinfo.m_numOutputNodes = s_grayScale ? 1 : 3; // Gray scale value of the pixel
    genCinfo.m_genomeCinfo.m_createBiasNode = true;
    genCinfo.m_genomeCinfo.m_networkType = NeuralNetworkType::GENERAL;
    genCinfo.m_genomeCinfo.m_activationProvider = &activationProvider;
    genCinfo.m_mutationParams.m_changeActivationRate = 0.03f;
    genCinfo.m_mutationParams.m_activationProvider = &activationProvider;
    genCinfo.m_fitnessCalculator = fitnessCalculator;
    genCinfo.m_generationParams.m_maxStagnantCount = 30;
    genCinfo.m_numThreads = 64;

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

        std::shared_ptr<GenomeBase> bestGenome = generation.getGenomesInFitnessOrder()[0].getGenome()->clone();
        float fitness = fitnessCalculator->calcFitness(bestGenome.get());
        std::cout << "Best Fitness: " << fitness << std::endl;
        std::cout << "Number of total nodes: " << bestGenome->getNumNodes() << std::endl;
        std::cout << "Number of enabled edges: " << bestGenome->getNumEnabledEdges() << std::endl;
        std::cout << "Number of species: " << generation.getAllSpecies().size() << std::endl;
        std::cout << "Best Species: " << generation.getAllSpeciesInBestFitnessOrder()[0].get() << std::endl;
        std::cout << "=============================" << std::endl;

        if (fitness > 240.f)
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
