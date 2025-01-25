#include <iostream>

// Include GLAD, GLFW, and ImGui
#include "OpenVDBLoader.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>


// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define NANOVDB_USE_OPENVDB
// Include project headers
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#include "Renderer.hpp"
#include "Shader.hpp"
#include "Utils/OpenToNano.hpp"

extern "C" void pointToGridFloat(HNS::OpenFloatGrid& in_data, float voxelSize, HNS::NanoFloatGrid& out_data, const cudaStream_t& stream);

// Callback function for window resizing
void framebuffer_size_callback(GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); }

// Camera variables
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 5.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

// Movement speed
float cameraSpeed = 5.0f;  // Adjust as needed

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Mouse input
float lastX = 400, lastY = 300;
float yaw = -90.0f;
float pitch = 0.0f;
bool firstMouse = true;

// Field of view
float fov = 45.0f;

// Process input
void processInput(GLFWwindow* window) {
	// Close window on 'Escape' key press
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

	float adjustedSpeed = cameraSpeed * deltaTime;

	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) cameraPos += adjustedSpeed * cameraFront;  // Move forward
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) cameraPos -= adjustedSpeed * cameraFront;  // Move backward
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * adjustedSpeed;  // Move left
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * adjustedSpeed;  // Move right
}

// Mouse callback
void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
	if (ImGui::GetIO().WantCaptureMouse) return;

	if (firstMouse) {
		lastX = (float)xpos;
		lastY = (float)ypos;
		firstMouse = false;
	}

	float xoffset = (float)(xpos - lastX);
	float yoffset = (float)(lastY - ypos);  // Reversed since y-coordinates go from bottom to top
	lastX = (float)xpos;
	lastY = (float)ypos;

	float sensitivity = 0.2f;  // Adjust sensitivity as needed
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	// Constrain the pitch
	if (pitch > 89.0f) pitch = 89.0f;
	if (pitch < -89.0f) pitch = -89.0f;

	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraFront = glm::normalize(front);
}

// Scroll callback
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	if (ImGui::GetIO().WantCaptureMouse) return;

	fov -= (float)yoffset;
	if (fov < 1.0f) fov = 1.0f;
	if (fov > 90.0f) fov = 90.0f;
}

int main() {
	// Initialize GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW\n";
		return -1;
	}

	// Configure GLFW (OpenGL version 3.3, Core Profile)
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


	// Create a windowed mode window and its OpenGL context
	GLFWwindow* window = glfwCreateWindow(800, 600, "HNanoViewer", nullptr, nullptr);
	if (!window) {
		std::cerr << "Failed to create GLFW window\n";
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Register callbacks
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// Capture the mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

	// Load OpenGL function pointers using GLAD
	if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
		std::cerr << "Failed to initialize GLAD\n";
		return -1;
	}

	// Initialize ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	// Setup ImGui IO
	ImGuiIO& io = ImGui::GetIO();
	(void)io;

	// Setup ImGui style
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 460 core");

	// Initialize OpenVDB Loader
	OpenVDBLoader vdbLoader;
	const std::string vdbFilename = "C:/Users/zphrfx/Desktop/bunny_cloud.vdb";

	// Volume data
	GLuint volumeTexture;
	bool vdbLoaded = false;


	// Initialize Renderer
	Renderer renderer;
	renderer.init();

	// Shader
	const Shader shader("C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/vertex_shader.vert",
	                    "C:/Users/zphrfx/Desktop/hdk/hdk_clion/HNanoSolver/src/HNanoViewer/shaders/fragment_shader.frag");


	// Performance metrics
	float frameTime = 0.0f;
	float fps = 0.0f;

	// Cube vertices (positions of a cube from -0.5 to 0.5 in all axes)
	constexpr float cubeVertices[] = {
	    // Positions
	    -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f,
	    -0.5f, -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, -0.5f, 0.5f,
	    -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,
	    0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,
	    -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f,
	    -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f,
	};

	unsigned int cubeVAO, cubeVBO;
	glGenVertexArrays(1, &cubeVAO);
	glGenBuffers(1, &cubeVBO);

	glBindVertexArray(cubeVAO);

	glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

	glBindVertexArray(0);

	float loadVDBTime = 0.0f;
	float gridDataTime = 0.0f;
	float advectionStepTime = 0.0f;
	float VDBToTextureTime = 0.0f;
	float totalLoadingTime = 0.0f;


	// Main loop
	while (!glfwWindowShouldClose(window)) {
		const auto currentFrame = static_cast<float>(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		// Calculate frame time and FPS
		frameTime = deltaTime * 1000.0f;  // Convert to milliseconds
		fps = 1.0f / deltaTime;

		// Input
		processInput(window);

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Create ImGui window
		{
			ImGui::Begin("Performance Metrics");

			// Rendering performance
			ImGui::Text("Frame Time: %.3f ms", frameTime);
			ImGui::Text("FPS: %.1f", fps);

			// Loading performance
			if (totalLoadingTime > 0.0f) {
				ImGui::Separator();
				ImGui::Text("Loading Times:");
				ImGui::Text("  Total Loading Time: %.3f ms", totalLoadingTime);
				ImGui::Text("  Load VDB Time: %.3f ms", loadVDBTime);
				ImGui::Text("  Grid Data Time: %.3f ms", gridDataTime);
				ImGui::Text("  Advection Step Time: %.3f ms", advectionStepTime);
				ImGui::Text("  VDBToTexture Time: %.3f ms", VDBToTextureTime);
			}

			ImGui::Separator();

			if (ImGui::Button("Load VDB File")) {
				// Measure loadVDB time
				auto startTime = std::chrono::high_resolution_clock::now();
				vdbLoader.loadVDB(vdbFilename);
				auto endTime = std::chrono::high_resolution_clock::now();
				loadVDBTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();

				cudaFree(0);
			}

			// try to ditch the cold start of the first cuda call

			if (ImGui::Button("Run Kernels")) {

				openvdb::GridBase::Ptr pBaseGrid = vdbLoader.getGridBase();
				if (!pBaseGrid) {
					std::cerr << "Error: Grid is not loaded." << std::endl;
					continue;
				}
				openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(pBaseGrid);

				auto totalStartTime = std::chrono::high_resolution_clock::now();

				std::chrono::time_point<std::chrono::steady_clock> startTime;
				std::chrono::time_point<std::chrono::steady_clock> endTime;


				HNS::OpenFloatGrid gridData;
				{
					startTime = std::chrono::high_resolution_clock::now();
					HNS::extractFromOpenVDB<openvdb::FloatGrid, float>(grid, gridData);
					endTime = std::chrono::high_resolution_clock::now();
					gridDataTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
				}

				// Measure Advection Step time
				HNS::NanoFloatGrid nanoGridData;
				{
					startTime = std::chrono::high_resolution_clock::now();
					pointToGridFloat(gridData, grid->voxelSize()[0], nanoGridData, nullptr);
					endTime = std::chrono::high_resolution_clock::now();
					advectionStepTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
				}


				// Measure VDBToTexture time
				startTime = std::chrono::high_resolution_clock::now();
				vdbLoaded = vdbLoader.VDBToTexture(volumeTexture, nanoGridData);
				endTime = std::chrono::high_resolution_clock::now();
				VDBToTextureTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();

				// End total timer
				auto totalEndTime = std::chrono::high_resolution_clock::now();
				totalLoadingTime = std::chrono::duration<float, std::milli>(totalEndTime - totalStartTime).count();
			}
		}

		ImGui::Text("VDB File: %s", vdbFilename.c_str());
		ImGui::End();

		// Rendering
		ImGui::Render();

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		// Camera setup
		glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
		glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 500.0f);

		// Render voxels if loaded
		// Render volume if loaded
		if (vdbLoaded) {
			glEnable(GL_DEPTH_TEST);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			shader.use();

			glm::mat4 model = glm::mat4(1.0f);
			shader.setMat4("model", model);
			shader.setMat4("view", view);
			shader.setMat4("projection", projection);
			shader.setVec3("cameraPos", cameraPos);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_3D, volumeTexture);
			shader.setInt("volumeTexture", 0);

			glBindVertexArray(cubeVAO);
			glDrawArrays(GL_TRIANGLES, 0, 36);
			glBindVertexArray(0);
		}

		// Draw ImGui content
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// Swap buffers and poll IO events
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// Cleanup ImGui
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	// Terminate GLFW
	glfwTerminate();
	return 0;
}
