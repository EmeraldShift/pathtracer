#include <cmath>

#include "scene.h"
#include "light.h"
#include "bvh.h"
#include "../ui/TraceUI.h"
#include <glm/gtx/extended_min_max.hpp>
#include <iostream>
#include <glm/gtx/io.hpp>

using namespace std;

bool Geometry::intersect(ray &r, isect &i) const {
    if (hasBoundingBoxCapability() && !(bounds.intersect(r))) {
        return false;
    }


    // Transform the ray into the object's local coordinate space
    auto pos = transform->globalToLocalCoords(r.getPosition());
    auto dir = transform->globalToLocalCoords(r.getPosition() + r.getDirection()) - pos;
    double length = glm::length(dir);
    dir = glm::normalize(dir);

    // Backup world pos/dir, switch to local pos/dir
    auto wPos = r.getPosition();
    auto wDir = r.getDirection();
    r.setPosition(pos);
    r.setDirection(dir);

    bool ret = false;
    if (intersectLocal(r, i)) {
        // Transform the intersection point & normal returned back into global space.
        i.setN(transform->localToGlobalCoordsNormal(i.getN()));
        i.setT(i.getT() / length);
        ret = true;
    }

    // Restore world pos/dir
    r.setPosition(wPos);
    r.setDirection(wDir);
    return ret;
}

//bool Geometry::intersect(ray& r, isect& i) const {
//	double tmin, tmax;
//	if (hasBoundingBoxCapability() && !(bounds.intersect(r, tmin, tmax))) return false;
//	// Transform the ray into the object's local coordinate space
//	glm::dvec3 pos = transform->globalToLocalCoords(r.getPosition());
//	glm::dvec3 dir = transform->globalToLocalCoords(r.getPosition() + r.getDirection()) - pos;
//	double length = glm::length(dir);
//	dir = glm::normalize(dir);
//	// Backup World pos/dir, and switch to local pos/dir
//	glm::dvec3 Wpos = r.getPosition();
//	glm::dvec3 Wdir = r.getDirection();
//	r.setPosition(pos);
//	r.setDirection(dir);
//	bool rtrn = false;
//	if (intersectLocal(r, i))
//	{
//		// Transform the intersection point & normal returned back into global space.
//		i.setN(transform->localToGlobalCoordsNormal(i.getN()));
//		i.setT(i.getT()/length);
//		rtrn = true;
//	}
//	// Restore World pos/dir
//	r.setPosition(Wpos);
//	r.setDirection(Wdir);
//	return rtrn;
//}

bool Geometry::hasBoundingBoxCapability() const {
    // by default, primitives do not have to specify a bounding box.
    // If this method returns true for a primitive, then either the computeBoundingBox() or
    // the ComputeLocalBoundingBox() method must be implemented.

    // If no bounding box capability is supported for an object, that object will
    // be checked against every single ray drawn.  This should be avoided whenever possible,
    // but this possibility exists so that new primitives will not have to have bounding
    // boxes implemented for them.
    return false;
}

void Geometry::computeBoundingBox() {
    // take the object's local bounding box, transform all 8 points on it,
    // and use those to find a new bounding box.

    BoundingBox localBounds = computeLocalBoundingBox();

    glm::dvec3 min = localBounds.getMin();
    glm::dvec3 max = localBounds.getMax();

    glm::dvec4 v, newMax, newMin;

    v = transform->localToGlobalCoords(glm::dvec4(min[0], min[1], min[2], 1));
    newMax = v;
    newMin = v;
    v = transform->localToGlobalCoords(glm::dvec4(max[0], min[1], min[2], 1));
    newMax = glm::max(newMax, v);
    newMin = glm::min(newMin, v);
    v = transform->localToGlobalCoords(glm::dvec4(min[0], max[1], min[2], 1));
    newMax = glm::max(newMax, v);
    newMin = glm::min(newMin, v);
    v = transform->localToGlobalCoords(glm::dvec4(max[0], max[1], min[2], 1));
    newMax = glm::max(newMax, v);
    newMin = glm::min(newMin, v);
    v = transform->localToGlobalCoords(glm::dvec4(min[0], min[1], max[2], 1));
    newMax = glm::max(newMax, v);
    newMin = glm::min(newMin, v);
    v = transform->localToGlobalCoords(glm::dvec4(max[0], min[1], max[2], 1));
    newMax = glm::max(newMax, v);
    newMin = glm::min(newMin, v);
    v = transform->localToGlobalCoords(glm::dvec4(min[0], max[1], max[2], 1));
    newMax = glm::max(newMax, v);
    newMin = glm::min(newMin, v);
    v = transform->localToGlobalCoords(glm::dvec4(max[0], max[1], max[2], 1));
    newMax = glm::max(newMax, v);
    newMin = glm::min(newMin, v);

    bounds.setMax(glm::dvec3(newMax));
    bounds.setMin(glm::dvec3(newMin));
}

Scene::Scene() = default;

Scene::~Scene() = default;

void Scene::add(Geometry *obj) {
    obj->computeBoundingBox();
    sceneBounds.merge(obj->getBoundingBox());
    objects.emplace_back(obj);
}

void Scene::add(Light *light) {
    lights.emplace_back(light);
}

bool Scene::intersect(ray &r, isect &i) {
    bool have_one = bvh.traverse(r, i);

//    for (const auto &o : objects) {
//        if (o->intersect(r, i)) {
//            return true;
//        }
//    }

    // Fix point lights: transform into small object
    for (const auto &l : lights) {
        auto dist = l->getDistanceFrom(r.getPosition());

        // We assume all point lights are balls with radius 1
        auto minDot = dist
                / std::sqrt(1 + dist * dist);
        auto dir = glm::normalize(l->getDirection(r.getPosition()));
        auto dot = glm::dot(dir, glm::normalize(r.getDirection()));
        auto pow = ((dot - minDot) / (1 - minDot)) * ((dot - minDot) / (1 - minDot));
        if (dot > minDot) {
            // Close enough
            if (!have_one || (dist < i.getT())) {
                i.setMaterial(Material(l->getColor(), glm::dvec3(), glm::dvec3(), glm::dvec3(), glm::dvec3(), glm::dvec3(), 0, 0));
                i.setT(dist);
                have_one = true;
            }
        }
    }

    // if debugging,
    // if (TraceUI::m_debug)
    //     intersectCache.emplace_back(new ray(r), new isect(i));
    return have_one;
}

// Get any intersection with an object.  Return information about the 
// intersection through the reference parameter.
//bool Scene::intersect(ray &r, isect &i) const {
//    double tmin = 0.0;
//    double tmax = 0.0;
//    bool have_one = false;
//    for (const auto &obj : objects) {
//        isect cur;
//        if (obj->intersect(r, cur)) {
//            if (!have_one || (cur.getT() < i.getT())) {
//                i = cur;
//                have_one = true;
//            }
//        }
//    }
//    if (!have_one)
//        i.setT(1000.0);
//    // if debugging,
//    if (TraceUI::m_debug)
//        intersectCache.push_back(std::make_pair(new ray(r), new isect(i)));
//    return have_one;
//}

TextureMap *Scene::getTexture(string name) {
    auto itr = textureCache.find(name);
    if (itr == textureCache.end()) {
        textureCache[name].reset(new TextureMap(name));
        return textureCache[name].get();
    }
    return itr->second.get();
}


