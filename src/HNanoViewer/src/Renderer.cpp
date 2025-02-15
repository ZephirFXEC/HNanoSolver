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

	if (boundingBoxVAO)
		glDeleteVertexArrays(1, &boundingBoxVAO);
	if (boundingBoxVBO)
		glDeleteBuffers(1, &boundingBoxVBO);
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

void Renderer::render(const Shader& shader, GLuint volumeTexture, const glm::vec3& cameraPos, const glm::mat4& view, const glm::mat4& projection, const glm::mat4& model) const {
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	shader.use();

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


void Renderer::drawBoundingBox(const Shader& shader, const glm::vec3& min, const glm::vec3& max,
                               const glm::mat4& view, const glm::mat4& projection, const glm::mat4& model) {
    // Compute the eight corners of the bounding box.
    glm::vec3 v0(min.x, min.y, min.z);
    glm::vec3 v1(max.x, min.y, min.z);
    glm::vec3 v2(max.x, max.y, min.z);
    glm::vec3 v3(min.x, max.y, min.z);
    glm::vec3 v4(min.x, min.y, max.z);
    glm::vec3 v5(max.x, min.y, max.z);
    glm::vec3 v6(max.x, max.y, max.z);
    glm::vec3 v7(min.x, max.y, max.z);

    // Define vertices for the 12 line segments (24 vertices) of the box.
    float boxVertices[] = {
        // Bottom face
        v0.x, v0.y, v0.z,  v1.x, v1.y, v1.z,
        v1.x, v1.y, v1.z,  v2.x, v2.y, v2.z,
        v2.x, v2.y, v2.z,  v3.x, v3.y, v3.z,
        v3.x, v3.y, v3.z,  v0.x, v0.y, v0.z,

        // Top face
        v4.x, v4.y, v4.z,  v5.x, v5.y, v5.z,
        v5.x, v5.y, v5.z,  v6.x, v6.y, v6.z,
        v6.x, v6.y, v6.z,  v7.x, v7.y, v7.z,
        v7.x, v7.y, v7.z,  v4.x, v4.y, v4.z,

        // Vertical edges
        v0.x, v0.y, v0.z,  v4.x, v4.y, v4.z,
        v1.x, v1.y, v1.z,  v5.x, v5.y, v5.z,
        v2.x, v2.y, v2.z,  v6.x, v6.y, v6.z,
        v3.x, v3.y, v3.z,  v7.x, v7.y, v7.z,
    };

    // Create VAO/VBO if not already done.
    if (boundingBoxVAO == 0) {
        glGenVertexArrays(1, &boundingBoxVAO);
        glGenBuffers(1, &boundingBoxVBO);
    }

    glBindVertexArray(boundingBoxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, boundingBoxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(boxVertices), boxVertices, GL_DYNAMIC_DRAW);

    // Set vertex attribute for position.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Increase line width for better visibility.
    glLineWidth(1.0f);

    // Use the shader dedicated for drawing the bounding box.
    shader.use();
    shader.setMat4("model", model);
    shader.setMat4("view", view);
    shader.setMat4("projection", projection);

    // Draw the bounding box as lines.
    glDrawArrays(GL_LINES, 0, 24);

    // Optionally, reset the line width (if needed).
    glLineWidth(1.0f);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}