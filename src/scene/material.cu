#include "material.h"
#include "../ui/TraceUI.h"
#include "ray.h"

#include <algorithm>
#include "../fileio/images.h"

TextureMap::TextureMap(const string& filename) {
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