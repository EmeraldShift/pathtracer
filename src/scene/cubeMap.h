#pragma once

#include <memory>
#include "../vec.h"
#include <glm/vec3.hpp>

class TextureMap;
class ray;

class CubeMap {
	std::unique_ptr<TextureMap> tMap[6];
public:
	CubeMap();
	~CubeMap();

	void setNthMap(int n, TextureMap* m);

	f4 getColor(const ray& r) const;
};
