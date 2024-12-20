//
// Created by zphrfx on 02/12/2024.
//

#include "Renderer.hpp"

#include <glad/glad.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

Renderer::Renderer() : cubeVAO(0), cubeVBO(0), instanceVBO(0) {}

Renderer::~Renderer() {
	// Cleanup
	glDeleteVertexArrays(1, &cubeVAO);
	glDeleteBuffers(1, &cubeVBO);
	glDeleteBuffers(1, &instanceVBO);
}

void Renderer::init() { setupCube(); }

void Renderer::setupCube() {
	// Cube vertices (36 vertices for 12 triangles)
	constexpr float cubeVertices[] = {
	    // positions
	    -0.5f, -0.5f, -0.5f,  // Back face
	    0.5f,  -0.5f, -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f,

	    -0.5f, -0.5f, 0.5f,  // Front face
	    0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, -0.5f, 0.5f,

	    -0.5f, 0.5f,  0.5f,  // Left face
	    -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,

	    0.5f,  0.5f,  0.5f,  // Right face
	    0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,

	    -0.5f, -0.5f, -0.5f,  // Bottom face
	    0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f, -0.5f, 0.5f,  -0.5f, -0.5f, -0.5f,

	    -0.5f, 0.5f,  -0.5f,  // Top face
	    0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  -0.5f, 0.5f,  0.5f,  -0.5f, 0.5f,  -0.5f,
	};

	glGenVertexArrays(1, &cubeVAO);
	glGenBuffers(1, &cubeVBO);

	glBindVertexArray(cubeVAO);

	// Cube vertices
	glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

	// Position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// Unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Renderer::render(const Shader& shader, const std::vector<glm::vec3>& voxelPositions, const glm::mat4& view, const glm::mat4& projection) {
	// Update instance VBO
	if (instanceVBO == 0) {
		glGenBuffers(1, &instanceVBO);
	}
	glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
	glBufferData(GL_ARRAY_BUFFER, voxelPositions.size() * sizeof(glm::vec3), &voxelPositions[0], GL_STATIC_DRAW);

	// Configure instance attribute
	glBindVertexArray(cubeVAO);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glVertexAttribDivisor(1, 1);  // Tell OpenGL this is an instanced vertex attribute.
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// Render cubes
	shader.use();
	shader.setMat4("view", view);
	shader.setMat4("projection", projection);
	glm::mat4 model = glm::mat4(1.0f);
	shader.setMat4("model", model);

	glBindVertexArray(cubeVAO);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 36, voxelPositions.size());
	glBindVertexArray(0);
}
