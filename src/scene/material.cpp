#include "material.h"
#include "../ui/TraceUI.h"
#include "light.h"
#include "ray.h"

extern TraceUI *traceUI;

#include <glm/gtx/io.hpp>
#include <iostream>
#include "../fileio/images.h"

using namespace std;
extern bool debugMode;

Material::~Material() {
}

TextureMap::TextureMap(string filename) {
    data = readImage(filename.c_str(), width, height);
    if (data.empty()) {
        width = 0;
        height = 0;
        string error("Unable to load texture map '");
        error.append(filename);
        error.append("'.");
        throw TextureMapException(error);
    }
}

glm::dvec3 TextureMap::getMappedValue(const glm::dvec2 &coord) const {
    double x = coord[0] * width;
    double y = coord[1] * height;

    double lerp_x = x - (int) x;
    double lerp_y = y - (int) y;

    auto vlerp_xl = lerp_x * getPixelAt((int) x + 1, (int) y) + (1 - lerp_x) * getPixelAt((int) x, (int) y);
    auto vlerp_xr = lerp_x * getPixelAt((int) x + 1, (int) y + 1) + (1 - lerp_x) * getPixelAt((int) x, (int) y + 1);
    return lerp_y * vlerp_xr + (1 - lerp_y) * vlerp_xl;
}

glm::dvec3 TextureMap::getPixelAt(int x, int y) const {
    x = std::max(0, std::min(width - 1, x));
    y = std::max(0, std::min(height - 1, y));
    auto idx = (x + y * width) * 3;
    return glm::dvec3(data[idx] / 256.0, data[idx + 1] / 256.0, data[idx + 2] / 256.0);
}

glm::dvec3 MaterialParameter::value(const isect &is) const {
    if (0 != _textureMap)
        return _textureMap->getMappedValue(is.getUVCoordinates());
    else
        return _value;
}

double MaterialParameter::intensityValue(const isect &is) const {
    if (0 != _textureMap) {
        glm::dvec3 value(
                _textureMap->getMappedValue(is.getUVCoordinates()));
        return (0.299 * value[0]) + (0.587 * value[1]) +
               (0.114 * value[2]);
    } else
        return (0.299 * _value[0]) + (0.587 * _value[1]) +
               (0.114 * _value[2]);
}
