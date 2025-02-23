// Application.cpp

// Standard & third-party includes
#include "Renderer.hpp"

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "Application.hpp"

#include <iostream>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>
#include <vector>
// Your own headers
#include "OpenVDBLoader.hpp"
#include "Shader.hpp"
#include "Utils/GridBuilder.hpp"



extern "C" {
	extern "C" void accessBrick(const BrickMap& brickMap);
	extern "C" void advect(const BrickMap& brickMap, float dt);
	void Create(VolumeTexture* volumeTexture, const BrickMap* brickMap);
	void Update(VolumeTexture* volumeTexture, const BrickMap* brickMap);
}
// -------------------------------------------------------------
// Constructor & Destructor
// -------------------------------------------------------------
Application::Application()
    : window_(nullptr),
      cameraPos_(0, 0, 0),
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
      renderer_(nullptr),
      shader_(nullptr),
      wireframe_(nullptr),
      vdbLoaded_(false),
      vdbFilename_("C:/Users/zphrfx/Desktop/bunny_cloud.vdb") {}

BrickMap brickMap(Dim(8, 8, 8));
std::vector<std::pair<nanovdb::Coord, nanovdb::Coord>> bboxes;
std::vector<std::pair<nanovdb::Coord, float>> coordValue;


Application::~Application() { cleanup(); }

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
	renderer_ = new Renderer();
	renderer_->init();
	shader_ = new Shader("C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/vertex_shader.vert",
	                     "C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/fragment_shader.frag");
	wireframe_ = new Shader("C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/vertex_shader.vert",
	                        "C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/wireframe.frag");

	//Create(&volumeTex_, &brickMap);

	return true;
}

// -------------------------------------------------------------
// Input & Update
// -------------------------------------------------------------
void Application::processInput() {
	if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window_, true);

	float adjustedSpeed = cameraSpeed_ * deltaTime_;

	if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) cameraPos_ += adjustedSpeed * cameraFront_;
	if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) cameraPos_ -= adjustedSpeed * cameraFront_;
	if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) cameraPos_ -= glm::normalize(glm::cross(cameraFront_, cameraUp_)) * adjustedSpeed;
	if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) cameraPos_ += glm::normalize(glm::cross(cameraFront_, cameraUp_)) * adjustedSpeed;
	if (glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS) cameraPos_ += glm::vec3(0, 1, 0) * adjustedSpeed;
	if (glfwGetKey(window_, GLFW_KEY_E) == GLFW_PRESS) cameraPos_ -= glm::vec3(0, 1, 0) * adjustedSpeed;
}

void Application::update() {
	// Update timing
	float currentFrame = static_cast<float>(glfwGetTime());
	deltaTime_ = currentFrame - lastFrame_;
	lastFrame_ = currentFrame;

	bboxes.clear();


	advect(brickMap, 1.0f/24.0f);



	Voxel* v1 = brickMap.getBrickAtHost(BrickCoord(0, 0, 0));

	for (int i = 0; i < 16; i++) {
		printf("Brick 0 %d: %f\n", i, v1[i].density);
	}


	const std::vector<BrickCoord> brickCoords = brickMap.getActiveBricks();
	printf("Active bricks: %llu\n", brickCoords.size());

	for (const auto& brick : brickCoords) {
		nanovdb::Coord min = {brick[0], brick[1], brick[2]};
		nanovdb::Coord max = {brick[0] + 1, brick[1] + 1, brick[2] + 1};
		bboxes.emplace_back(min, max);
	}

	//Update(&volumeTex_, &brickMap);
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
		ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", cameraPos_.x, cameraPos_.y, cameraPos_.z);
		ImGui::End();
	}

	// Clear and configure OpenGL state
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	// Set up camera matrices
	glm::mat4 view = glm::lookAt(cameraPos_, cameraPos_ + cameraFront_, cameraUp_);
	glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 500.0f);

	for (const auto brickbbox : bboxes) {
		glm::vec3 min = glm::vec3(brickbbox.first[0], brickbbox.first[1], brickbbox.first[2]);
		glm::vec3 max = glm::vec3(brickbbox.second[0], brickbbox.second[1], brickbbox.second[2]);

		renderer_->drawBoundingBox(*wireframe_, min, max, view, projection, glm::mat4(1.0f));
	}

	/*
	shader_->use();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, volumeTex_.texture());

	shader_->setInt("volumeTexture", 0);
	*/


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
	if (!init()) return -1;

	if(!brickMap.allocateBrickAt({0,0,0})) {
		printf("Failed to allocate brick\n");
	}

	{
		ScopedTimer timer("BrickMap::KernelAccess");
		accessBrick(brickMap);
	}

	cudaDeviceSynchronize();

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

	if (window_) glfwDestroyWindow(window_);
	glfwTerminate();

	// Free dynamically allocated resources
	delete shader_;
	delete wireframe_;
	delete renderer_;
}

// -------------------------------------------------------------
// Static Callback Functions
// -------------------------------------------------------------
void Application::framebufferSizeCallback(GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); }

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
	float yoffset = app->lastY_ - static_cast<float>(ypos);  // y offset is reversed
	app->lastX_ = static_cast<float>(xpos);
	app->lastY_ = static_cast<float>(ypos);

	float sensitivity = 0.2f;
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	app->yaw_ += xoffset;
	app->pitch_ += yoffset;

	// Constrain the pitch angle
	if (app->pitch_ > 89.0f) app->pitch_ = 89.0f;
	if (app->pitch_ < -89.0f) app->pitch_ = -89.0f;

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
	if (app->fov_ < 1.0f) app->fov_ = 1.0f;
	if (app->fov_ > 90.0f) app->fov_ = 90.0f;
}
