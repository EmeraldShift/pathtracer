#include "trimesh.h"

#include "geometry.h"

#include <cmath>

Geometry *TrimeshFace::create(const Material &mat, Trimesh *parent, int a, int b, int c) {
    // Compute the face normal here, not on the fly
    auto a_coords = parent->vertices[a];
    auto b_coords = parent->vertices[b];
    auto c_coords = parent->vertices[c];

    auto vab = (b_coords - a_coords);
    auto vac = (c_coords - a_coords);
    auto vcb = (b_coords - c_coords);

    // If this is a degenerate face, quit
    if (f4m::length2(vab) == 0.0f
        || f4m::length2(vac) == 0.0f
        || f4m::length2(vcb) == 0.0f)
        return nullptr;

    auto face = TrimeshFace();
    auto normal = f4m::normalize(f4m::cross(vab, vac));

    // Quick f4-escape hack: we don't have matrices
    glm::vec3 glm_vab(vab[0], vab[1], vab[2]);
    glm::vec3 glm_vac(vac[0], vac[1], vac[2]);
    glm::vec3 glm_normal(normal[0], normal[1], normal[2]);
    auto inverse = glm::inverse(glm::mat3(glm_vab, glm_vac, glm_normal));

    face.materials[0] = mat;
    face.vertices[0] = a_coords;
    face.vertices[1] = b_coords;
    face.vertices[2] = c_coords;
    face.normal = normal;
    face.uInv = {inverse[0][0], inverse[1][0], inverse[2][0]};
    face.vInv = {inverse[0][1], inverse[1][1], inverse[2][1]};
    face.nInv = {inverse[0][2], inverse[1][2], inverse[2][2]};
    if (!parent->materials.empty()) {
        face.hasMaterials = true;
        face.materials[0] = parent->materials[a];
        face.materials[1] = parent->materials[b];
        face.materials[2] = parent->materials[c];
    }
    return new Geometry(face);
}

Trimesh::~Trimesh() {
    for (auto f : faces)
        delete f;
}

void Trimesh::addVertex(const f4 &v) {
    vertices.emplace_back(v);
}

void Trimesh::addMaterial(const Material &m) {
    materials.emplace_back(m);
}

// Returns false if the vertices a,b,c don't all exist
bool Trimesh::addFace(int a, int b, int c) {
    int vCount = vertices.size();

    if (a >= vCount || b >= vCount || c >= vCount)
        return false;

    auto face = TrimeshFace::create(mat, this, a, b, c);
    if (face)
        faces.push_back(face);

    // Don't add faces to the scene's object list so we can cull by bounding
    // box
    return true;
}

// Check to make sure that if we have per-vertex materials or normals
// they are the right number.
const char *Trimesh::doubleCheck() {
    if (!materials.empty() && materials.size() != vertices.size())
        return "Bad Trimesh: Wrong number of materials.";
    return nullptr;
}

__host__ __device__
bool TrimeshFace::intersect(const ray &r, isect &i) const {
    auto p = r.getPosition() - vertices[0];
    auto d = r.getDirection();
    auto p_z = f4m::dot(nInv, p);
    auto d_z = f4m::dot(nInv, d);
    if (p_z < 0 == d_z < 0)
        return false;

    auto t = -p_z / d_z;
    if (t > i.getT())
        return false;

    auto hit = p + t * d;
    auto u = f4m::dot(uInv, hit); // p1
    auto v = f4m::dot(vInv, hit); // p2;
    auto w = 1 - u - v; // p0

    if (u < 0 || u > 1 || v < 0 || v > 1 || u + v > 1)
        return false;

    i.setT(t);
    i.setN(normal);
    i.setUVCoordinates({u, v});
    if (hasMaterials) {
        Material m0 = w * materials[0];
        Material m1 = u * materials[1];
        Material m2 = v * materials[2];
        m0 += m1;
        m0 += m2;
        i.setMaterial(m0);
    } else {
        i.setMaterial(materials[0]);
    }
    return true;
}


Trimesh *Trimesh::addVertices(f4 a, f4 b, f4 c) {
    addVertex(a);
    addVertex(b);
    addVertex(c);
    return this;
}

Trimesh *Trimesh::addCube(f4 *v) {
    addVertices(v[0], v[1], v[2]);
    addVertices(v[3], v[2], v[1]);
    addVertices(v[4], v[6], v[5]);
    addVertices(v[7], v[5], v[6]);
    addVertices(v[0], v[2], v[4]);
    addVertices(v[6], v[4], v[2]);
    addVertices(v[1], v[5], v[3]);
    addVertices(v[7], v[3], v[5]);
    addVertices(v[0], v[1], v[4]);
    addVertices(v[5], v[4], v[1]);
    addVertices(v[2], v[6], v[3]);
    addVertices(v[7], v[3], v[6]);
    for (int i = 0; i < 12; i++)
        addFace(i * 3, i * 3 + 1, i * 3 + 2);
    return this;
}

Trimesh *Trimesh::fromBox(Material &mat, TransformNode *transform) {
    auto p000 = transform->localToGlobalCoords(glm::vec3(-0.5, -0.5, -0.5));
    auto p001 = transform->localToGlobalCoords(glm::vec3(-0.5, -0.5, 0.5));
    auto p010 = transform->localToGlobalCoords(glm::vec3(-0.5, 0.5, -0.5));
    auto p011 = transform->localToGlobalCoords(glm::vec3(-0.5, 0.5, 0.5));
    auto p100 = transform->localToGlobalCoords(glm::vec3(0.5, -0.5, -0.5));
    auto p101 = transform->localToGlobalCoords(glm::vec3(0.5, -0.5, 0.5));
    auto p110 = transform->localToGlobalCoords(glm::vec3(0.5, 0.5, -0.5));
    auto p111 = transform->localToGlobalCoords(glm::vec3(0.5, 0.5, 0.5));
    f4 vertices[8] = {p000, p001, p010, p011, p100, p101, p110, p111};
    return (new Trimesh(mat))->addCube(vertices);
}

Trimesh *Trimesh::fromSquare(Material &mat, TransformNode *transform) {
    auto p000 = transform->localToGlobalCoords(glm::vec3(-0.5, -0.5, -0.001));
    auto p001 = transform->localToGlobalCoords(glm::vec3(-0.5, -0.5, 0.001));
    auto p010 = transform->localToGlobalCoords(glm::vec3(-0.5, 0.5, -0.001));
    auto p011 = transform->localToGlobalCoords(glm::vec3(-0.5, 0.5, 0.001));
    auto p100 = transform->localToGlobalCoords(glm::vec3(0.5, -0.5, -0.001));
    auto p101 = transform->localToGlobalCoords(glm::vec3(0.5, -0.5, 0.001));
    auto p110 = transform->localToGlobalCoords(glm::vec3(0.5, 0.5, -0.001));
    auto p111 = transform->localToGlobalCoords(glm::vec3(0.5, 0.5, 0.001));
    f4 vertices[8] = {p000, p001, p010, p011, p100, p101, p110, p111};
    return (new Trimesh(mat))->addCube(vertices);
}