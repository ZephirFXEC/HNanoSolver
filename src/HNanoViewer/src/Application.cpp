// Application.cpp

// Standard & third-party includes
#include <iostream>
#include <cmath>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/matrix_transform.hpp>

// Your own headers
#include "OpenVDBLoader.hpp"
#include "Renderer.hpp"
#include "Shader.hpp"
#include "Utils/GridBuilder.hpp"

#include "Application.hpp"

#include "BrickMap/BrickMap.cuh"

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


	// Create a BrickMap with a maximum of 65,536 bricks.
	BrickMap brickMap(16132);
	brickMap.initialize();

	// Prepare some voxel updates.
	// Global coordinate space is 0..8191 in each axis.
	std::vector<VoxelUpdate> updates;
	updates.push_back({100, 200, 300, Voxel(69.0f, 420.0f, 0.0f, nanovdb::Vec3f(1.0f, 0.0f, 0.0f))});
	updates.push_back({150, 250, 350, Voxel(3.0f,  75.0f, 0.5f, nanovdb::Vec3f(1.0f, 1.0f, 0.0f))});
	updates.push_back({1023, 1023, 1023, Voxel(10.5f,  50.0f, 0.2f, nanovdb::Vec3f(0.0f, 1.0f, 0.0f))});


	// Build the brick map from the host array.
	brickMap.buildFromUpdates(updates);

	// Query some voxels.
	Voxel v1 = brickMap.queryVoxel(100, 200, 300);
	std::cout << "Voxel (100,200,300): density=" << v1.density << "\n";
	Voxel v2 = brickMap.queryVoxel(1023, 1023, 1023);
	std::cout << "Voxel (1024,1024,1024): density=" << v2.density << "\n";
	Voxel v4 = brickMap.queryVoxel(150, 250, 350);
	std::cout << "Voxel (150,250,350): density=" << v4.density << "\n";

	// Perform a dynamic update: remove the voxel (set to empty) at (150,250,350).
	brickMap.updateVoxel(150, 250, 350, Voxel());  // Voxel() is empty.

	// (Alternatively, after advecting a field on the GPU you might call deviceUpdateVoxel() from within another kernel.)

	// Optionally, run a cleanup pass to deallocate any bricks that have become empty.
	brickMap.cleanupEmptyBricks();

	Voxel v4_after = brickMap.queryVoxel(150, 250, 350);
	std::cout << "Voxel (150,250,350) after removal: density=" << v4_after.density << "\n";

	std::cout << "BrickMap operations completed successfully.\n";

    // Main loop
    while (!glfwWindowShouldClose(window_)) {
        update();
        processInput();
        render();
    }

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
