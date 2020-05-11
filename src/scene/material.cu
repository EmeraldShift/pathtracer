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

f4 TextureMap::getMappedValue(const float2 &coord) const {
    auto x = coord.x * width;
    auto y = coord.y * height;

    auto lerp_x = x - (int) x;
    auto lerp_y = y - (int) y;

    auto vlerp_xl = lerp_x * getPixelAt((int) x + 1, (int) y) + (1 - lerp_x) * getPixelAt((int) x, (int) y);
    auto vlerp_xr = lerp_x * getPixelAt((int) x + 1, (int) y + 1) + (1 - lerp_x) * getPixelAt((int) x, (int) y + 1);
    return lerp_y * vlerp_xr + (1 - lerp_y) * vlerp_xl;
}

f4 TextureMap::getPixelAt(int x, int y) const {
    x = std::max(0, std::min(width - 1, x));
    y = std::max(0, std::min(height - 1, y));
    auto idx = (x + y * width) * 3;
    return {data[idx] / 256.0f, data[idx + 1] / 256.0f, data[idx + 2] / 256.0f};
}
