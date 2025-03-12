// Application.hpp
#pragma once

#include "BrickMap.cuh"

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <string>



// Forward declarations for your own classes:
class Shader;
class Renderer;

class Application {
   public:
	Application();
	~Application();

	// Runs the main loop; returns exit code.
	int run();

   private:
	// Initialization and cleanup
	bool init();
	void cleanup();

	// Main loop helpers
	void processInput();
	void update();
	void render();

	// GLFW window and context
	GLFWwindow* window_;

	// Camera state
	glm::vec3 cameraPos_;
	glm::vec3 cameraFront_;
	glm::vec3 cameraUp_;
	float cameraSpeed_;

	// Timing
	float deltaTime_;
	float lastFrame_;

	// Mouse input state
	float lastX_;
	float lastY_;
	float yaw_;
	float pitch_;
	bool firstMouse_;

	// Field of view
	float fov_;

	// Components (these are created dynamically; adjust as needed)
	Renderer* renderer_;
	Shader* shader_;
	Shader* wireframe_;

	// Volume data state
	bool vdbLoaded_;

	// VDB file path (adjust as needed)
	std::string vdbFilename_;

	// Callback wrappers
	static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
	static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
};
