#include <iostream>
#include "cubeMap.h"
#include "ray.h"
#include "../ui/TraceUI.h"
#include <glm/gtx/io.hpp>

glm::vec3 CubeMap::getColor(const ray& r) const {

    // First, map a unit direction to the cube
    auto dir = glm::normalize(r.getDirection());
    int idx = 2;
    if (std::abs(dir[1]) > std::abs(dir[2]))
        idx = 1;
    if (std::abs(dir[0]) > std::abs(dir[2]) && std::abs(dir[0]) > std::abs(dir[1]))
        idx = 0;

    auto forward = glm::vec3((dir[idx] < 0 ? -1 : 1) * (idx == 0),
                              (dir[idx] < 0 ? -1 : 1) * (idx == 1),
                              (dir[idx] < 0 ? -1 : 1) * (idx == 2));
    dir /= glm::dot(dir, forward);

    // Then, map to the 6 textures (involves fixups for orientation, etc)
    idx *= 2;
    if (dir[idx / 2] < 0 != (idx == 4))
        idx++;

    auto up = idx / 2 == 1 ? glm::vec3(0, 0, (dir[idx / 2] < 0 ? -1 : 1)) : glm::vec3(0, 1, 0);
    auto right = glm::normalize(glm::cross(forward, up));
    auto uv = glm::dvec2(0.5f + glm::dot(dir, right) / 2, 0.5f + glm::dot(dir, up) / 2);
    return tMap[idx]->getMappedValue(uv);
}

CubeMap::CubeMap() = default;

CubeMap::~CubeMap() = default;

void CubeMap::setNthMap(int n, TextureMap *m) {
    if (m != tMap[n].get())
        tMap[n].reset(m);
}
