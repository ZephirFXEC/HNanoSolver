//
// Created by zphrfx on 02/12/2024.
//

#pragma once

#include <glad/glad.h>

#include <glm/glm.hpp>

#include "Shader.hpp"

class Renderer {
   public:
	Renderer();
	~Renderer();

	// Initialize rendering resources
	void init();

	// Render the scene
	void render(const Shader& shader, GLuint volumeTexture, const glm::vec3& cameraPos, const glm::mat4& view, const glm::mat4& projection, const glm::mat4& model) const;
	void drawBoundingBox(const Shader& shader, const glm::vec3& min, const glm::vec3& max,
						  const glm::mat4& view, const glm::mat4& projection, const glm::mat4& model);
   private:
	// Cube mesh data
	unsigned int cubeVAO, cubeVBO, instanceVBO;
	GLuint boundingBoxVAO;
	GLuint boundingBoxVBO;
	// Setup cube mesh
	void setupCube();
};
