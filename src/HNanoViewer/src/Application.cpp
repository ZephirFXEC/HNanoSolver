// Application.cpp

// Standard & third-party includes
#include <iostream>
#include <cmath>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
// Your own headers
#include "OpenVDBLoader.hpp"
#include "Renderer.hpp"
#include "Shader.hpp"
#include "Utils/GridBuilder.hpp"

#include "Application.hpp"
#include "BrickMap.cuh"

extern "C" void accessBrick(const BrickMap& brickMap);

// -------------------------------------------------------------
// Constructor & Destructor
// -------------------------------------------------------------
Application::Application()
    : window_(nullptr),
      cameraPos_(5.0f, 0.0f, 0.0f),
      cameraFront_(0.0f, 0.0f, -1.0f),
      cameraUp_(0.0f, 1.0f, 0.0f),
      cameraSpeed_(5.0f),
      deltaTime_(0.0f),
      lastFrame_(0.0f),
      lastX_(400.0f),
      lastY_(300.0f),
      yaw_(-90.0f),
      pitch_(0.0f),
      firstMouse_(true),
      fov_(45.0f),
      vdbLoader_(nullptr),
      renderer_(nullptr),
      shader_(nullptr),
	  wireframe_(nullptr),
      vdbLoaded_(false),
      volumeTexture_(0),
      vdbFilename_("C:/Users/zphrfx/Desktop/bunny_cloud.vdb")
{
}

Application::~Application() {
    cleanup();
}

// -------------------------------------------------------------
// Initialization
// -------------------------------------------------------------
bool Application::init() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    // Configure GLFW for OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    window_ = glfwCreateWindow(800, 600, "HNanoViewer", nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);

    // Set the pointer to this instance for callbacks
    glfwSetWindowUserPointer(window_, this);

    // Register callbacks
    glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
    glfwSetCursorPosCallback(window_, mouseCallback);
    glfwSetScrollCallback(window_, scrollCallback);

    // Set input mode (change as desired)
    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // Load OpenGL function pointers using GLAD
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "Failed to initialize GLAD\n";
        return false;
    }

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 460 core");

    // Initialize your components
    vdbLoader_ = new OpenVDBLoader();
    renderer_ = new Renderer();
    renderer_->init();
    shader_ = new Shader(
        "C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/vertex_shader.vert",
        "C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/fragment_shader.frag"
    );
	wireframe_ = new Shader(
		"C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/vertex_shader.vert",
		"C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/wireframe.frag"
	);

    return true;
}

// -------------------------------------------------------------
// Input & Update
// -------------------------------------------------------------
void Application::processInput() {
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window_, true);

    float adjustedSpeed = cameraSpeed_ * deltaTime_;

    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos_ += adjustedSpeed * cameraFront_;
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos_ -= adjustedSpeed * cameraFront_;
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos_ -= glm::normalize(glm::cross(cameraFront_, cameraUp_)) * adjustedSpeed;
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos_ += glm::normalize(glm::cross(cameraFront_, cameraUp_)) * adjustedSpeed;
	if (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS)
		cameraPos_ += glm::vec3(0, 1, 0) * adjustedSpeed;
	if (glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS)
		cameraPos_ -= glm::vec3(0, 1, 0) * adjustedSpeed;
}

void Application::update() {
    // Update timing
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime_ = currentFrame - lastFrame_;
    lastFrame_ = currentFrame;
}

// -------------------------------------------------------------
// Rendering
// -------------------------------------------------------------
void Application::render() {
	// Start the Dear ImGui frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	openvdb::math::BBox<openvdb::Vec3d> bbox;
    // ImGui window for performance metrics and controls
    {
        float frameTime = deltaTime_ * 1000.0f;
        float fps = 1.0f / deltaTime_;
        ImGui::Begin("Performance Metrics");
        ImGui::Text("Frame Time: %.3f ms", frameTime);
        ImGui::Text("FPS: %.1f", fps);

        // VDB load/run buttons
        if (ImGui::Button("Load VDB File"))
            vdbLoader_->loadVDB(vdbFilename_);
        if (ImGui::Button("Run Kernels")) {
            auto pBaseGrid = vdbLoader_->getGridBase();
            if (!pBaseGrid) {
                std::cerr << "Error: Grid is not loaded." << std::endl;
            } else {
                auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(pBaseGrid);
                HNS::GridIndexedData gridData;
                HNS::IndexGridBuilder<openvdb::FloatGrid> indexGridBuilder(grid, &gridData);
                indexGridBuilder.addGrid(grid, "density");
                indexGridBuilder.build();

                vdbLoaded_ = vdbLoader_->VDBToTexture(volumeTexture_, &gridData, bbox);
            }
        }
        ImGui::Text("VDB File: %s", vdbFilename_.c_str());
        ImGui::End();
    }

    // Clear and configure OpenGL state
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    // Set up camera matrices
    glm::mat4 view = glm::lookAt(cameraPos_, cameraPos_ + cameraFront_, cameraUp_);
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 500.0f);

    // Render if the volume texture is ready
    if (vdbLoaded_) {

    	glm::vec3 worldMin(bbox.min().x(), bbox.min().y(), bbox.min().z());
    	glm::vec3 worldMax(bbox.max().x(), bbox.max().y(), bbox.max().z());
    	glm::vec3 size = worldMax - worldMin;
    	glm::vec3 center = (worldMin + worldMax) / 2.0f;
    	glm::mat4 modelMatrix = glm::mat4(1.0f); // glm::translate(glm::mat4(1.0f), center) * glm::scale(glm::mat4(1.0f), size);

	    renderer_->render(*shader_, volumeTexture_, cameraPos_, view, projection, modelMatrix);
    	renderer_->drawBoundingBox(*wireframe_, glm::vec3(-1,-1,-1), glm::vec3(1,1,1), view, projection, modelMatrix);

    }
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window_);
    glfwPollEvents();
}



// -------------------------------------------------------------
// Main Loop
// -------------------------------------------------------------
int Application::run() {
    if (!init())
        return -1;

	BrickMap brickMap(256, 256, 256);

	if (!brickMap.allocateBrickAt(0,0,0)) {
		fprintf(stderr, "Failed to allocate brick at (%i,%i,%i)\n", 0,0,0);
	}

	if (!brickMap.allocateBrickAt(1,1,1)) {
		fprintf(stderr, "Failed to allocate brick at (%i,%i,%i)\n", 1,0,0);
	}

	{
		ScopedTimer timer("BrickMap::Kernel");
    	accessBrick(brickMap);
	}

	Voxel* brick = brickMap.getBrickAtHost(0,0,0);
	Voxel* topBrick = brickMap.getBrickAtHost(1,1,1);

	Voxel* v1 = &brick[0];
	Voxel* v1b = &brick[32*32*32-1];
	Voxel* v2 = &topBrick[0];
	Voxel* v3 = &topBrick[-1];

	printf("Brick %u: density Voxel 0 = %u\n", 0, v1->density);
	printf("Brick %u: density Voxel 32767 = %u\n", 0, v1b->density);
	printf("Brick %u: density Voxel 0 = %u\n", 1, v2->density);
	printf("Brick %u: density Voxel -1 = %u\n", 1, v3->density);

	std::cout << "BrickMap operations completed successfully.\n";
    /*// Main loop
    while (!glfwWindowShouldClose(window_)) {
        update();
        processInput();
        render();
    }*/


    return 0;
}

// -------------------------------------------------------------
// Cleanup
// -------------------------------------------------------------
void Application::cleanup() {
    // Shutdown ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (window_)
        glfwDestroyWindow(window_);
    glfwTerminate();

    // Free dynamically allocated resources
    delete shader_;
	delete wireframe_;
    delete renderer_;
    delete vdbLoader_;
}

// -------------------------------------------------------------
// Static Callback Functions
// -------------------------------------------------------------
void Application::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void Application::mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    Application* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (app->firstMouse_) {
        app->lastX_ = static_cast<float>(xpos);
        app->lastY_ = static_cast<float>(ypos);
        app->firstMouse_ = false;
    }

    float xoffset = static_cast<float>(xpos) - app->lastX_;
    float yoffset = app->lastY_ - static_cast<float>(ypos); // y offset is reversed
    app->lastX_ = static_cast<float>(xpos);
    app->lastY_ = static_cast<float>(ypos);

    float sensitivity = 0.2f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    app->yaw_ += xoffset;
    app->pitch_ += yoffset;

    // Constrain the pitch angle
    if (app->pitch_ > 89.0f)
        app->pitch_ = 89.0f;
    if (app->pitch_ < -89.0f)
        app->pitch_ = -89.0f;

    glm::vec3 front;
    front.x = cos(glm::radians(app->yaw_)) * cos(glm::radians(app->pitch_));
    front.y = sin(glm::radians(app->pitch_));
    front.z = sin(glm::radians(app->yaw_)) * cos(glm::radians(app->pitch_));
    app->cameraFront_ = glm::normalize(front);
}

void Application::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    Application* app = reinterpret_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;
    if (ImGui::GetIO().WantCaptureMouse) return;

    app->fov_ -= static_cast<float>(yoffset);
    if (app->fov_ < 1.0f)
        app->fov_ = 1.0f;
    if (app->fov_ > 90.0f)
        app->fov_ = 90.0f;
}
