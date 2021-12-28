/*
* PBDTest main.cpp
*
* Copyright (C) 2021 Kohei Nagasawa All Rights Reserved.
*/

#include <iostream>
#include <memory>
#include <thread>

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

#include <Physics/World/World.h>
#include <Physics/Systems/PointBasedSystem.h>
#include <Physics/Solvers/PBD/PBDSolver.h>
#include <Geometry/Shapes/PlaneShape.h>

using namespace kPhysLib;

class MySystem : public kBaseSystem
{
public:

    MySystem(const int argc, const char** argv)
        : kBaseSystem(argc, argv)
    {
        kWindowCommand::registerAllCommands();
        m_scene = std::make_shared<kRenderScene>();

        // Create and initialize scene
        initialize("PBD Test");
    }

    kBool run() override
    {
        bool result = true;

        while (1)
        {
            // Step the world
            if (m_running)
            {
                m_world.step(m_deltaTime);
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

        return result;
    }

    void handleInput();
    void init();
    void initScene();
    void initRenderer();
    void postStep();

    // For simulation
    World m_world;
    PointBasedSystem m_pointBasedSystem;
    float m_deltaTime = 1.0f/60.f;

    bool m_running = true;

    std::shared_ptr<kRenderScene> m_scene;

    std::vector<kRenderScene::ObjectId> m_vertexObjectIds;

    kBasicDebugViewer* m_debugViewer;
};

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

void MySystem::handleInput()
{
    if (kInputModule::getInstance().getGlobalKeyLog().isKeyDown('P'))
    {
        m_running = !m_running;
    }
}

void MySystem::initRenderer()
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
            std::shared_ptr<kDirectionalLight> dl = std::make_shared<kDirectionalLight>(kColor::MAGENTA, kVector4(0, -1, 1), false);
            m_scene->addLight(dl, kTransform_Identity);
        }

        // Directional light
        {
            std::shared_ptr<kDirectionalLight> dl = std::make_shared<kDirectionalLight>(kColor::CYAN, kVector4(-1, -1, -1), false);
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

namespace
{
    void createCube(int& numVerts, std::vector<Vector4>& positions, PointBasedSystem::Cinfo& cinfo)
    {
        numVerts = 8;
        positions.reserve(numVerts);
        positions.push_back(Vector4(-1.f, -1.f, -1.f));
        positions.push_back(Vector4(-1.f, -1.f, 1.f));
        positions.push_back(Vector4(1.f, -1.f, 1.f));
        positions.push_back(Vector4(1.f, -1.f, -1.f));
        positions.push_back(Vector4(-1.f, 1.f, -1.f));
        positions.push_back(Vector4(-1.f, 1.f, 1.f));
        positions.push_back(Vector4(1.f, 1.f, 1.f));
        positions.push_back(Vector4(1.f, 1.f, -1.f));

        Vector4 translation(0.f, 3.f, 0.f);
        for (int i = 0; i < numVerts; i++)
        {
            positions[i] += translation;
        }

        cinfo.m_vertexPositions.reserve(numVerts);
        for (int i = 0; i < numVerts; i++)
        {
            cinfo.m_vertexPositions.push_back(positions[i]);
        }
        float stiffness1 = 0.2f;
        float stiffness2 = 0.05f;
        cinfo.m_vertexConnectivity.reserve(12);
        cinfo.m_vertexConnectivity.push_back({ 0, 1, stiffness1 });
        cinfo.m_vertexConnectivity.push_back({ 1, 2, stiffness1 });
        cinfo.m_vertexConnectivity.push_back({ 2, 3, stiffness1 });
        cinfo.m_vertexConnectivity.push_back({ 3, 0, stiffness1 });
        cinfo.m_vertexConnectivity.push_back({ 4, 5, stiffness1 });
        cinfo.m_vertexConnectivity.push_back({ 5, 6, stiffness1 });
        cinfo.m_vertexConnectivity.push_back({ 6, 7, stiffness1 });
        cinfo.m_vertexConnectivity.push_back({ 7, 4, stiffness1 });
        cinfo.m_vertexConnectivity.push_back({ 0, 4, stiffness2 });
        cinfo.m_vertexConnectivity.push_back({ 1, 5, stiffness2 });
        cinfo.m_vertexConnectivity.push_back({ 2, 6, stiffness2 });
        cinfo.m_vertexConnectivity.push_back({ 3, 7, stiffness2 });
        cinfo.m_mass = (float)numVerts;
    }

    void createSphere(int& numVerts, std::vector<Vector4>& positions, PointBasedSystem::Cinfo& cinfo)
    {
        kGeomData geom = kGeometryUtil::createSphere(kVector4_Zero, 2.f, 8, 4);
        numVerts = geom.m_vertices.size();
        positions.reserve(numVerts + 1);
        cinfo.m_vertexPositions.reserve(numVerts + 1);

        Vector4 avgPos = Vec4_0;
        for (const kVector4& v : geom.m_vertices)
        {
            Vector4 vv = kvToV(v);
            vv(1) += 2.0f;
            positions.push_back(vv);
            avgPos += vv;
            cinfo.m_vertexPositions.push_back(positions.back());
        }

        avgPos /= SimdFloat((float)numVerts);
        cinfo.m_vertexPositions.push_back(avgPos);
        positions.push_back(avgPos);

        float stiffness = .5f;
        for (const kGeomData::Triangle& t : geom.m_triangles)
        {
            if (t.m_a != t.m_b)
            {
                cinfo.m_vertexConnectivity.push_back({ (int)t.m_a, (int)t.m_b, stiffness });
            }
            if (t.m_b != t.m_c)
            {
                cinfo.m_vertexConnectivity.push_back({ (int)t.m_b, (int)t.m_c, stiffness });
            }
            if (t.m_a != t.m_c)
            {
                cinfo.m_vertexConnectivity.push_back({ (int)t.m_c, (int)t.m_a, stiffness });
            }
        }

        for (int i = 0; i < numVerts; i++)
        {
            cinfo.m_vertexConnectivity.push_back({ i, numVerts, stiffness });
        }

        cinfo.m_mass = (float)(numVerts + 1);
        numVerts++;
    }

    void createTeapot(int& numVerts, std::vector<Vector4>& positions, PointBasedSystem::Cinfo& cinfo)
    {
        kGeomData geom;
        kObjSerializer::load("D:/user/Kohei/development/projects/PhysLib/0.00.1/resources/models/teapot.obj", geom);

        numVerts = geom.m_vertices.size();
        positions.reserve(numVerts);
        cinfo.m_vertexPositions.reserve(numVerts);
        for (const kVector4& v : geom.m_vertices)
        {
            Vector4 vv = kvToV(v);
            vv *= SimdFloat(0.1f);
            vv(1) += 1.0f;
            positions.push_back(vv);
            cinfo.m_vertexPositions.push_back(positions.back());
        }

        float stiffness = .25f;
        for (const kGeomData::Triangle& t : geom.m_triangles)
        {
            if (t.m_a != t.m_b)
            {
                cinfo.m_vertexConnectivity.push_back({ (int)t.m_a, (int)t.m_b, stiffness });
            }
            if (t.m_b != t.m_c)
            {
                cinfo.m_vertexConnectivity.push_back({ (int)t.m_b, (int)t.m_c, stiffness });
            }
            if (t.m_a != t.m_c)
            {
                cinfo.m_vertexConnectivity.push_back({ (int)t.m_c, (int)t.m_a, stiffness });
            }
        }

        cinfo.m_mass = (float)numVerts;
    }
}

void MySystem::initScene()
{
    // Create a cube
    {
        PointBasedSystem::Cinfo cinfo;
        cinfo.m_solverIterations = 4;
        cinfo.m_dampingFactor = 0.2f;
        cinfo.m_radius = 0.15f;
        //cinfo.m_solverType = PointBasedSystemSolver::Type::MASS_SPRING;

        int numVerts;
        std::vector<Vector4> positions;

        //createCube(numVerts, positions, cinfo);
        createSphere(numVerts, positions, cinfo);
        //createTeapot(numVerts, positions, cinfo);

        m_pointBasedSystem.init(cinfo);

        kMaterialId mRedId = createMaterial("RED", kColor::RED);
        for (int i = 0; i < numVerts; i++)
        {
            // Create sphere
            kGeomData geom = kGeometryUtil::createSphere(kVector4_Zero, 0.15f, 4, 2);

            // Add the geometry to the scene
            std::shared_ptr<kRenderMesh> mesh = std::make_shared<kRenderMesh>(geom, kTransform_Identity, mRedId, kObjectMotionType::MOVABLE);
            kScaleTransform st; st.setTranslation(vTokV(positions[i]));
            m_vertexObjectIds.push_back(m_scene->addMeshObject(mesh, st));
        }
    }

    // Create ground
    {
        std::shared_ptr<Shape> groundShape = std::make_shared<PlaneShape>(Vector4(0.f, 1.f, 0.f, 0.f));
        m_pointBasedSystem.addCollider(groundShape);

        // Add render geometry
        {
            kMaterialId mGrayId = createMaterial("GRAY", kColor::GRAY);

            kScaleTransform st;
            {
                kGeomData geom = kGeometryUtil::createTriangle(kVector4(-1000.f, 0.f, -1000.f), kVector4(1000.f, 0.f, 1000.f), kVector4(1000.f, 0.f, -1000.f));
                std::shared_ptr<kRenderMesh> mesh = std::make_shared<kRenderMesh>(geom, kTransform_Identity, mGrayId, kObjectMotionType::STATIC);
                m_scene->addMeshObject(mesh, st);
            }
            {
                kGeomData geom = kGeometryUtil::createTriangle(kVector4(-1000.f, 0.f, -1000.f), kVector4(-1000.f, 0.f, 1000.f), kVector4(1000.f, 0.f, 1000.f));
                std::shared_ptr<kRenderMesh> mesh = std::make_shared<kRenderMesh>(geom, kTransform_Identity, mGrayId, kObjectMotionType::STATIC);
                m_scene->addMeshObject(mesh, st);
            }
        }
    }

    m_world.addSystem(m_pointBasedSystem);
}

void MySystem::init()
{
    initScene();
    initRenderer();
}

void MySystem::postStep()
{
    const std::vector<Vector4>& positions = m_pointBasedSystem.getVertexPositions();
    const int numVerts = (int)positions.size();
    for (int i = 0; i < numVerts; i++)
    {
        kTransform transform;
        transform.setTranslation(vTokV(positions[i]));
        m_scene->setObjectTransform(m_vertexObjectIds[i], transform);
    }

    // Draw edges
    {
        const PointBasedSystem::Vertices& vertices = m_pointBasedSystem.getVertices();
        const PointBasedSystem::Edges& edges = m_pointBasedSystem.getEdges();
        int currentVertexIdx = 0;
        const PointBasedSystem::Vertex* currentVertex = &vertices[currentVertexIdx];
        kVector4 start = vTokV(positions[currentVertexIdx]);
        for (int i = 0; i < (int)edges.size(); i++)
        {
            if ((currentVertex->m_edgeStart + currentVertex->m_numEdges) == i)
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

int main(const int argc, const char** argv)
{
    std::unique_ptr<MySystem> sys = std::make_unique<MySystem>(argc, argv);
    K_ASSERT(0xd8e81f94, sys, "Failed to initialize base system.");

    sys->init();
    sys->run();
    sys->quit();

    return 0;
}

