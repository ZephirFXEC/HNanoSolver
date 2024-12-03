//
// Created by zphrfx on 02/12/2024.
//

#pragma once

#include <glm/glm.hpp>
#include <string>

class Shader {
   public:
	unsigned int ID;

	// Constructor reads and builds the shader
	Shader(const std::string& vertexPath, const std::string& fragmentPath);

	// Use/activate the shader
	void use() const;

	// Utility uniform functions
	void setBool(const std::string& name, bool value) const;
	void setInt(const std::string& name, int value) const;
	void setFloat(const std::string& name, float value) const;
	void setMat4(const std::string& name, const glm::mat4& mat) const;
	void setVec3(const std::string& name, const glm::vec3& value) const;

};
