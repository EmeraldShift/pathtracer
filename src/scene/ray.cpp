#include "ray.h"
#include "../ui/TraceUI.h"
#include "scene.h"


const Material &isect::getMaterial() const {
    return material ? *material : obj->getMaterial();
}

ray::ray(const glm::dvec3 &pos,
         const glm::dvec3 &dir,
         const glm::dvec3 &atten,
         RayType type)
        : pos(pos), dir(dir), invdir(invert(dir)), atten(atten), type(type) {
    TraceUI::addRay(ray_thread_id);
}

ray::ray(const ray &other) : pos(other.pos), dir(other.dir), atten(other.atten) {
    TraceUI::addRay(ray_thread_id);
}

ray::~ray() {
}

ray &ray::operator=(const ray &other) {
    pos = other.pos;
    dir = other.dir;
    invdir = other.invdir;
    atten = other.atten;
    type = other.type;
    return *this;
}

glm::dvec3 ray::at(const isect &i) const {
    return at(i.getT());
}

thread_local unsigned int ray_thread_id = 0;
