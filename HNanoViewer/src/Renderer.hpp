//
// Created by zphrfx on 02/12/2024.
//

#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "Shader.hpp"

class Renderer {
   public:
	Renderer();
	~Renderer();

	// Initialize rendering resources
	void init();

	// Render the scene
	void render(const Shader& shader, const std::vector<glm::vec3>& voxelPositions, const glm::mat4& view, const glm::mat4& projection);

   private:
	// Cube mesh data
	unsigned int cubeVAO, cubeVBO, instanceVBO;

	// Setup cube mesh
	void setupCube();
};
