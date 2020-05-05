#include "ray.h"
#include "../ui/TraceUI.h"
#include "scene.h"

ray::ray(const glm::dvec3 &pos,
         const glm::dvec3 &dir)
        : pos(pos), dir(dir), invdir(invert(dir)) {
    TraceUI::addRay(ray_thread_id);
}

ray::ray(const ray &other) : pos(other.pos), dir(other.dir) {
    TraceUI::addRay(ray_thread_id);
}

ray::~ray() {
}

ray &ray::operator=(const ray &other) {
    pos = other.pos;
    dir = other.dir;
    invdir = other.invdir;
    return *this;
}

glm::dvec3 ray::at(const isect &i) const {
    return at(i.getT());
}

thread_local unsigned int ray_thread_id = 0;
