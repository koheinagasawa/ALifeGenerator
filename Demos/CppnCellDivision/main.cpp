/*
* CppnCellDivision main.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <memory>
#include <algorithm>

//
// Includes for renderer
//

#include <Common/Base/kBase.h>
#include <Common/Base/Object/kSingleton.h>
#include <Common/Base/System/kBaseSystem.h>
#include <Common/Base/Types/kColors.h>
#include <Common/Base/Types/kColorUtil.h>
#include <Common/Base/Serialize/ObjFile/kObjSerializer.h>
#include <Common/Application/Input/kInputModule.h>
#include <Common/Application/Window/Platform/Windows/kWindowFactoryWin.h>
#include <Common/Application/Window/Command/kWindowCommand.h>
#include <Graphics/Graphics/System/kGraphicsModule.h>
#include <Graphics/DX11/System/kDX11GraphicsSystem.h>
#include <Graphics/Graphics/Scene/kRenderScene.h>
#include <Graphics/Graphics/Scene/kRenderSceneTypes.h>
#include <Graphics/Graphics/Scene/Light/Directional/kDirectionalLight.h>
#include <Graphics/Graphics/Material/Data/kMaterial.h>
#include <Graphics/Graphics/Render/Pipeline/Rasterizer/Forward/kForwardRenderingPipeline.h>
#include <Graphics/Graphics/Shading/ShadingManager/kSingleShadingManager.h>
#include <Graphics/Graphics/Shading/Library/kShaderLibrary.h>
#include <Graphics/Graphics/View/Camera/kCamera.h>
#include <Geometry/Geometry/Data/kGeomData.h>
#include <Geometry/Geometry/Util/kGeometryUtil.h>
#include <Graphics/Graphics/View/Camera/Control/Input/kSwitchableInputCameraController.h>
#include <Graphics/Graphics/DebugDisplay/Viewers/BasicDebugViewer/kBasicDebugViewer.h>

//
// Includes for simulation
//

#include <Common/PseudoRandom.h>
#include <Physics/World/World.h>
#include <Physics/Systems/PointBasedSystem.h>
#include <Physics/Solvers/PBD/PBDSolver.h>
#include <Geometry/Shapes/PlaneShape.h>

//
// Include for EvoAlgo
//

#include <EvoAlgo/GeneticAlgorithms/Base/GenomeBase.h>
#include <EvoAlgo/NeuralNetwork/NeuralNetworkFactory.h>
#include <EvoAlgo/CppnCellDivision/CppnCellCreature.h>
#include <EvoAlgo/GeneticAlgorithms/Base/Activations/ActivationProvider.h>

using namespace kPhysLib;

namespace
{
    kMaterialId createMaterial(const char* name, kColor color)
    {
        std::shared_ptr<kMaterial> material = std::make_shared<kMaterial>(name);
        {
            kVector4 col; kColorUtil::colorToVector4(color, col);
            material->addValueData<kVector4>(kMaterialParamType::DIFFUSE, col);
            col.mul(0.2f);
            material->addValueData<kVector4>(kMaterialParamType::ALBEDO, col);
            kColorUtil::colorToVector4(kColor::WHITE, col);
            col(3) = 0.00001f; // w component of specular is roughness of specular.
            material->addValueData<kVector4>(kMaterialParamType::SPECULAR, col);
        }
        return kMaterialLibrary::getInstance().registerMaterial(material);
    }

    kVector4 vTokV(const Vector4& v)
    {
        return kVector4(v(0), v(1), v(2));
    }

    Vector4 kvToV(const kVector4& v)
    {
        return Vector4(v(0), v(1), v(2));
    }
}

class MySystem;

class CppnDivision
{
public:
    static CppnDivision& getInstance() { return s_instance; }

    void init(MySystem* system, int seed)
    {
        m_system = system;

        m_world = std::make_unique<World>();

        //float stiffness = 0.25f;
        float stiffness = 0.1f;
        //float stiffness = 0.05f;

        {
            PointBasedSystem::Cinfo cinfo;
            cinfo.m_solverIterations = 4;
            //cinfo.m_dampingFactor = 0.002f;
            cinfo.m_dampingFactor = 0.1f;
            cinfo.m_radius = 0.15f;
            cinfo.m_vertexPositions.push_back(Vector4(-0.15f, 0.f, 0.f));
            cinfo.m_vertexPositions.push_back(Vector4(0.15f, 0.f, 0.f));
            cinfo.m_vertexConnectivity.push_back(PointBasedSystem::Cinfo::Connection{ 0, 1, stiffness });
            cinfo.m_mass = 1.0f;
            cinfo.m_gravity = Vec4_0;

            m_simulation = std::make_shared<PointBasedSystem>();
            m_simulation->subscribeToOnParticleAdded(onCellAdded);
            m_simulation->init(cinfo);
            //{
            //    PBD::Solver* solver = static_cast<PBD::Solver*>(m_simulation->getSolver().get());
            //    solver->setDampingType(PBD::Solver::VelocityDampingType::SHAPE_MATCH);
            //}

            m_world->addSystem(*m_simulation.get());
        }

        m_activationProvider = std::make_shared<DefaultActivationProvider>([](float value) { return 1.f / (1.f + expf(-4.9f * value)); });

        {
            CppnCellCreature::Cinfo cinfo;
            cinfo.m_simulation = m_simulation;
            cinfo.m_connectionStiffness = stiffness;
            cinfo.m_numMaxCells = 2000;
            cinfo.m_divisionInterval = 300;

            // Create a network
            {
                using Network = GenomeBase::Network;
                using Node = GenomeBase::Node;
                using Edge = GenomeBase::Edge;

                constexpr int numInputNodes = (int)CppnCellCreature::InputNode::NUM_INPUT_NODES;
                constexpr int numOutputNodes = (int)CppnCellCreature::OutputNode::NUM_OUTPUT_NODES;

                // Calculate the number of nodes
                int numNodes = numInputNodes + numOutputNodes + 1; // +1 is for bias node

                constexpr int numHiddenLayers = 2;
                for (int i = 0; i < numHiddenLayers; i++)
                {
                    numNodes += numInputNodes;
                }

                // Create buffers
                Network::Nodes nodes;
                Network::Edges edges;
                Network::NodeIds outputNodes;
                Network::NodeIds inputNodes;

                // Create nodes
                int nodeId = 0;
                nodes.reserve(numNodes);
                inputNodes.reserve(numInputNodes);
                outputNodes.reserve(numOutputNodes);

                // Create input nodes.
                for (int i = 0; i < numInputNodes; i++)
                {
                    NodeId id = nodeId++;
                    nodes.insert({ id, Node(Node::Type::INPUT) });
                    inputNodes.push_back(id);
                }

                // Create hidden nodes
                for (int i = 0; i < numHiddenLayers; i++)
                {
                    for (int j = 0; j < numInputNodes; j++)
                    {
                        NodeId id = nodeId++;
                        nodes.insert({ id, Node(Node::Type::HIDDEN) });
                        nodes[id].setActivation(m_activationProvider->getActivation());
                    }
                }

                // Create output nodes
                for (int i = 0; i < numOutputNodes; i++)
                {
                    NodeId id = nodeId++;
                    nodes.insert({ id, Node(Node::Type::OUTPUT) });
                    nodes[id].setActivation(m_activationProvider->getActivation());
                    outputNodes.push_back(id);
                }

                // Create a bias node
                NodeId biasNode;
                constexpr int biasNodeValue = 1.0f;
                {
                    biasNode = nodeId++;
                    nodes.insert({ biasNode, Node(Node::Type::BIAS) });
                    nodes[biasNode].setValue(biasNodeValue);
                }

                // Create fully connected network
                int edgeId = 0;
                int startL1Node = 0;
                for (int layer = 0; layer < numHiddenLayers + 1; layer++)
                {
                    int numL1Nodes = numInputNodes;
                    int numL2Nodes = (layer == numHiddenLayers) ? numOutputNodes : numInputNodes;
                    int startL2Node = startL1Node + numL1Nodes;

                    // Fully connect each layer
                    for (int i = 0, inNode = startL1Node; i < numL1Nodes; i++, inNode++)
                    {
                        for (int j = 0, outNode = startL2Node; j < numL2Nodes; j++, outNode++)
                        {
                            edges.insert({ EdgeId(edgeId++), Edge(inNode, outNode) });
                        }
                    }

                    // Create edges from the bias node
                    if (layer > 0)
                    {
                        for (int j = 0, outNode = startL2Node; j < numL2Nodes; j++, outNode++)
                        {
                            edges.insert({ EdgeId(edgeId++), Edge(biasNode, outNode) });
                        }
                    }

                    startL1Node += numL1Nodes;
                }

                // Randomize edge weights
                auto randomGenerator = std::make_shared<PseudoRandom>(seed);
                for (auto& edge : edges)
                {
                    edge.second.setWeight(randomGenerator->randomReal(-5.0f, 5.0f));
                }

                GenomeBase::NetworkPtr network = std::make_shared<Network>(nodes, edges, inputNodes, outputNodes);
                m_genome = std::make_unique<GenomeBase>(network, biasNode);
                cinfo.m_genome = m_genome.get();
            }

            m_creature = std::make_unique<CppnCellCreature>(cinfo);
            m_world->addSystem(*m_creature.get());
        }
    }

    void step()
    {
        m_world->step(1.0f / 60.f);
    }

    const CppnCellCreature& getCreature() const { return *m_creature; }

    void clear()
    {
        m_world = nullptr;
        m_simulation = nullptr;
        m_creature = nullptr;
        m_activationProvider = nullptr;
    }

    static void onCellAdded(const std::vector<Vector4>& cellPositions);

    std::unique_ptr<World> m_world;
    std::shared_ptr<PointBasedSystem> m_simulation;
    std::unique_ptr<CppnCellCreature> m_creature;
    std::unique_ptr<GenomeBase> m_genome;
    std::shared_ptr<DefaultActivationProvider> m_activationProvider;
    MySystem* m_system;

    static CppnDivision s_instance;
};

CppnDivision CppnDivision::s_instance;

class MySystem : public kBaseSystem
{
public:

    MySystem()
        : kBaseSystem(0, nullptr)
    {
        kWindowCommand::registerAllCommands();
        m_scene = std::make_shared<kRenderScene>();

        // select object
        {
            kCommandParser::Command command("r");
            command.addValueType(kCommandParser::Integer); // id
            kWindowCommand::registerCommand(command, &resetCommand);
        }

        // Create and initialize scene
        initialize("CPPN Cell Division");

        m_materialIds[0] = createMaterial("RED", kColor::RED);
        m_materialIds[1] = createMaterial("PURPLE", kColor::PURPLE);
        m_materialIds[2] = createMaterial("BLUE", kColor::BLUE);
        m_materialIds[3] = createMaterial("CYAN", kColor::CYAN);
        m_materialIds[4] = createMaterial("GREEN", kColor::GREEN);
        m_materialIds[5] = createMaterial("LIME", kColor::LIME);
        m_materialIds[6] = createMaterial("YELLOW", kColor::YELLOW);
        m_materialIds[7] = createMaterial("ORANGE", kColor::ORANGE);
    }

    kBool run() override
    {
        std::thread workerThread(&MySystem::workerFunc, this);
        bool result = true;

        while (1)
        {
            if (m_reset)
            {
                reset(m_seed);
                m_reset = false;
            }

            // Step the world
            if (!m_pause)
            {
                CppnDivision::getInstance().step();
            }

            postStep();

            if (!stepAllModules())
            {
                result = false;
                break;
            }

            if (kWindowModule::getInstance().getNumRegisteredWindows() == 0) break;

            handleInput();
        }
        m_end = true;
        workerThread.join();

        return result;
    }

    void init()
    {
        CppnDivision::getInstance().init(this, m_seed);
        initRenderer();
    }

    void workerFunc()
    {
        while (!m_end)
        {
            char input[128];
            std::cin.getline(input, 128);
            kString inputString(input);
            int seed = std::atoi(inputString.c_str());
            kWindowCommand::addCommand(inputString);
            kWindowCommand::dispatchCommand();
        }
    }

    void initRenderer()
    {
        addModule(&kGraphicsModule::getInstance());

        // Initialize a graphics system.
        kGraphicsModule::getInstance().initialize(kGraphicsAPIType::DX11);

        // Create shader data base.
        kShaderLibrary::getInstance().createDataBase();

        // Add directional light
        {
            // Directional light
            {
                std::shared_ptr<kDirectionalLight> dl = std::make_shared<kDirectionalLight>(kColor::WHITE, kVector4(0, -1, 1), false);
                m_scene->addLight(dl, kTransform_Identity);
            }

            // Directional light
            {
                std::shared_ptr<kDirectionalLight> dl = std::make_shared<kDirectionalLight>(kColor::WHITE, kVector4(-1, -1, -1), false);
                m_scene->addLight(dl, kTransform_Identity);
            }
        }

        // Create a rendering pipeline
        std::shared_ptr<kRenderingPipeline> pipeline = std::make_shared<kForwardRenderingPipeline>();

        // Create a shading manager
        std::shared_ptr<kShadingManager> shadingManager = std::make_shared<kSingleShadingManager>(
            m_scene.get(),
            pipeline,
            createMaterial("DUMMY", kColor::WHITE),
            kVertexBufferGeomData::POSITION | kVertexBufferGeomData::NORMAL
            );
        m_scene->registerCallbackShadingManager(shadingManager);

        // Create windows
        {
            kWindowFactoryWin factory;
            auto& windowModule = kWindowModule::getInstance();

            kVector4 lookAt(0, 0, 0);

            auto winHandle = windowModule.createWindowInstance("PBD Test", &factory);
            windowModule.setSize(winHandle, 1200, 800);

            kVector4 pos(0.0f, 3.0f, 10.0f);
            auto camera = kCamera::createCamera(pos, lookAt);
            {
                kSwitchableInputCameraController* camCtl = new kSwitchableInputCameraController();
                camCtl->addObservingWindow(winHandle);
                camera->setController(std::unique_ptr<kCameraController>(std::move(camCtl)));
            }

            kGraphicsSystem* gSys = kGraphicsModule::getInstance().getGraphicsSystemRW();
            auto id = gSys->createViewPort(camera, m_scene, pipeline, shadingManager);
            gSys->assignViewPortToWindow(winHandle, id);
            gSys->enableDebugDisplayMode(id);

            {
                std::shared_ptr<kDebugViewer> viewer = kDebugViewerRegistry::getInstance().getViewer("Basic");
                m_debugViewer = static_cast<kBasicDebugViewer*>(viewer.get());
            }

            windowModule.showWindow(winHandle);
        }
    }

    void handleInput()
    {
        const kKeyLog& log = kInputModule::getInstance().getGlobalKeyLog();
        if (log.isKeyPressed('P'))
        {
            m_pause = !m_pause;
        }
        else if (log.isKeyPressed('R'))
        {
            m_reset = true;
        }
    }

    static void resetCommand(const kWindowCommand::CommandOutput& out)
    {
        kArray<kInt> ints = out.findValue<kInt>();
        getInstance().m_reset = true;
        getInstance().m_seed = ints[0];
    }

    void reset(int seed)
    {
        for (auto oid : m_vertexObjectIds)
        {
            m_scene->removeObject(oid);
        }
        m_vertexObjectIds.clear();

        m_colorCounter = 0;

        CppnDivision& cd = CppnDivision::getInstance();
        cd.clear();
        cd.init(this, seed);
    }

    void postStep()
    {
        CppnDivision& cd = CppnDivision::getInstance();
        const PointBasedSystem& pbs = *cd.m_simulation.get();
        const std::vector<Vector4>& positions = pbs.getVertexPositions();
        const int numVerts = (int)positions.size();
        for (int i = 0; i < numVerts; i++)
        {
            kTransform transform;
            transform.setTranslation(vTokV(positions[i]));
            m_scene->setObjectTransform(m_vertexObjectIds[i], transform);
        }

        // Draw edges
        {
            const PointBasedSystem::Vertices& vertices = pbs.getVertices();
            const PointBasedSystem::Edges& edges = pbs.getEdges();
            int currentVertexIdx = 0;
            if (vertices.size() > 0)
            {
                const PointBasedSystem::Vertex* currentVertex = &vertices[currentVertexIdx];
                kVector4 start = vTokV(positions[currentVertexIdx]);
                for (int i = 0; i < (int)edges.size(); i++)
                {
                    while ((currentVertex->m_edgeStart + currentVertex->m_numEdges) <= i)
                    {
                        currentVertexIdx++;
                        currentVertex = &vertices[currentVertexIdx];
                        start = vTokV(positions[currentVertexIdx]);
                    }

                    const PointBasedSystem::Edge& edge = edges[i];
                    kVector4 end = vTokV(positions[edge.m_otherVertex]);
                    m_debugViewer->drawLine(start, end, kColor::RED);
                }
            }
        }
    }

    void onCellAdded(const std::vector<Vector4>& cellPositions)
    {
        const int numCells = (int)cellPositions.size();
        for (int i = 0; i < numCells; i++)
        {
            // Create sphere
            kGeomData geom = kGeometryUtil::createSphere(kVector4_Zero, 0.15f, 4, 2);

            // Add the geometry to the scene
            std::shared_ptr<kRenderMesh> mesh = std::make_shared<kRenderMesh>(geom, kTransform_Identity, m_materialIds[m_colorCounter % NUM_COLORS], kObjectMotionType::MOVABLE);
            kScaleTransform st; st.setTranslation(vTokV(cellPositions[i]));
            m_vertexObjectIds.push_back(m_scene->addMeshObject(mesh, st));
        }

        m_colorCounter++;

        m_pause = true;
    }

    static MySystem& getInstance()
    {
        if (!s_instance)
        {
            s_instance = std::make_unique<MySystem>();
        }
        return *s_instance;
    }

    std::shared_ptr<kRenderScene> m_scene;

    std::vector<kRenderScene::ObjectId> m_vertexObjectIds;

    kBasicDebugViewer* m_debugViewer;

    bool m_pause = false;
    bool m_end = false;
    bool m_reset = false;
    int m_seed = 0;

    int m_colorCounter = 0;
    static constexpr int NUM_COLORS = 8;
    kMaterialId m_materialIds[NUM_COLORS];

    static std::unique_ptr<MySystem> s_instance;
};

std::unique_ptr<MySystem> MySystem::s_instance;

void CppnDivision::onCellAdded(const std::vector<Vector4>& cellPositions)
{
    getInstance().m_system->onCellAdded(cellPositions);
}

int main(const int argc, const char** argv)
{
    MySystem& sys = MySystem::getInstance();

    sys.init();
    sys.run();
    sys.quit();

    return 0;
}

