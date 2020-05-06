#include <cmath>

#include "scene.h"
#include "../SceneObjects/sphere.h"
#include <iostream>

using namespace std;

Scene::Scene() = default;

Scene::~Scene() = default;


void Scene::add(Sphere *sphere) {
    spheres.emplace_back(sphere);
}

void Scene::add(Trimesh *mesh) {
    for (const auto &f : mesh->getFaces())
        triangles.emplace_back(f);
}

bool Scene::intersect(ray &r, isect &i) {
    return sBvh.traverse(r, i) | tBvh.traverse(r, i);
}

TextureMap *Scene::getTexture(string name) {
    auto itr = textureCache.find(name);
    if (itr == textureCache.end()) {
        textureCache[name].reset(new TextureMap(name));
        return textureCache[name].get();
    }
    return itr->second.get();
}